"""Extract the text from the file."""
import mailbox
from json import dumps
from pathlib import Path
from typing import Dict, List

from config import PDF_MIN_WORDS

from docx import Document

from odf import text as odf_text
from odf.opendocument import load as odf_load

import pymupdf


class DocumentExtractor:
    """Extract text from a document."""

    def __init__(self, processed_dir: Path):
        """Initialize the attributes."""
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf(self, pdf_path: Path) -> Dict:
        """Extract text from PDF, skip if appears to be scanned."""
        doc = pymupdf.open(pdf_path)

        # Check first page for text content
        first_page_text = doc[0].get_text()
        word_count = len(first_page_text.split())

        if word_count < PDF_MIN_WORDS:
            doc.close()
            return None

        text = ''
        for page in doc:
            text += page.get_text()

        pages = len(doc)
        doc.close()

        return {
            'source': str(pdf_path),
            'type': 'pdf',
            'text': text.strip(),
            'metadata': {'pages': pages}
        }

    def extract_text_file(self, txt_path: Path) -> Dict:
        """Extract text from plain text file."""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        return {
            'source': str(txt_path),
            'type': 'text',
            'text': text.strip(),
            'metadata': {}
        }

    def extract_docx(self, docx_path: Path) -> Dict:
        """Extract text from DOCX."""
        doc = Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])

        return {
            'source': str(docx_path),
            'type': 'docx',
            'text': text.strip(),
            'metadata': {}
        }

    def extract_odt(self, odt_path: Path) -> Dict:
        """Extract text from ODT."""
        doc = odf_load(str(odt_path))
        paragraphs = doc.getElementsByType(odf_text.P)
        text = '\n'.join([str(p) for p in paragraphs])

        return {
            'source': str(odt_path),
            'type': 'odt',
            'text': text.strip(),
            'metadata': {}
        }

    def clean_email_body(self, body: str) -> str:
        """Remove quoted replies and excess whitespace."""
        lines = body.split('\n')
        cleaned = []

        for line in lines:
            # Skip common quote markers
            if (
                line.startswith('>')
                or (
                    line.startswith('On ')
                    and 'wrote:' in line)
            ):
                break
            cleaned.append(line)

        return '\n'.join(cleaned).strip()

    def extract_mbox(self, mbox_path: Path) -> List[Dict]:
        """Extract messages from mbox file."""
        mbox = mailbox.mbox(str(mbox_path))
        messages = []

        for idx in range(len(mbox)):
            try:
                message = mbox[idx]
            except Exception as e:
                print(f'Exception extracting message {idx}: {e}')
                continue

            try:
                subject = message.get('Subject', 'No Subject')
                sender = message.get('From', 'Unknown')
                date = message.get('Date', '')

                body = ''
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == 'text/plain':
                            body = part.get_payload(decode=True).decode(
                                'utf-8',
                                errors='ignore'
                            )
                            break
                else:
                    body = message.get_payload(decode=True).decode(
                        'utf-8',
                        errors='ignore'
                    )

                body = self.clean_email_body(body)

                if body:  # Only include messages with content
                    messages.append({
                        'source': f'{mbox_path.name}::{idx}',
                        'type': 'email',
                        'text': body,
                        'metadata': {
                            'subject': subject,
                            'from': sender,
                            'date': date
                        }
                    })
            except Exception as e:
                print(f'Error processing message {idx} in {mbox_path}: {e}')
                continue

        return messages

    def process_all(self, raw_dir: Path):
        """Process all documents and save to processed directory."""
        all_docs = []

        # Process PDFs
        print('Processing PDFs...')
        for pdf_path in (raw_dir / 'pdfs').glob('*.pdf'):
            doc = self.extract_pdf(pdf_path)
            if doc:
                all_docs.append(doc)

        print('Processing text files...')
        for txt_path in (raw_dir / 'docs').glob('*.txt'):
            all_docs.append(self.extract_text_file(txt_path))

        print('Processing DOCX files...')
        for docx_path in (raw_dir / 'docs').glob('*.docx'):
            try:
                all_docs.append(self.extract_docx(docx_path))
            except Exception as e:
                print(f'Error processing {docx_path}: {e}')

        print('Processing ODT files...')
        for odt_path in (raw_dir / 'docs').glob('*.odt'):
            try:
                all_docs.append(self.extract_odt(odt_path))
            except Exception as e:
                print(f'Error processing {odt_path}: {e}')

        print('Processing mbox files...')
        for mbox_path in (raw_dir / 'mbox').glob('*.mbox'):
            messages = self.extract_mbox(mbox_path)
            all_docs.extend(messages)

        output_file = self.processed_dir / 'extracted_documents.jsonl'
        with open(output_file, 'w') as f:
            for doc in all_docs:
                f.write(dumps(doc) + '\n')

        print(f'Processed {len(all_docs)} documents -> {output_file}')
        return len(all_docs)
