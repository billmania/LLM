"""Extract the text from the file."""
import mailbox
import unicodedata
from json import dumps
from pathlib import Path
from re import sub as substitute
from typing import Dict, List

from config import DOCUMENTS_FILE

from docx import Document

from odf import text as odf_text
from odf.opendocument import load as odf_load

import pymupdf4llm


def clean_text(text: str) -> str:
    """Remove problematic characters that confuse LLMs."""
    original_length = len(text)

    text = unicodedata.normalize('NFKD', text)

    # Replace common math symbols with text equivalents
    replacements = {
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        '∑': 'sum',
        '∫': 'integral',
        '√': 'sqrt',
        '∞': 'infinity',
        '∂': 'partial',
        '∆': 'delta',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'θ': 'theta',
        'λ': 'lambda',
        'μ': 'mu',
        'σ': 'sigma',
        'π': 'pi',
    }
    for symbol, replacement in replacements.items():
        text = text.replace(symbol, replacement)

    # Remove non-printable characters and control characters
    text = ''.join(
        char for char in text if unicodedata.category(char)[0] != 'C'
    )

    # Keep only ASCII plus common punctuation
    text = ''.join(
        char if ord(char) < 128 or char in '""''—–' else ' ' for char in text
    )

    text = substitute(r'\s+', ' ', text)
    text = text.strip()

    cleaned_length = len(text)
    print(
        'clean_text:'
        f' Removed {original_length - cleaned_length}'
        ' characters'
    )

    return text


class DocumentExtractor:
    """Extract text from a collection of documents.

    The results are written to the single file extracted_documents.jsonl.
    """

    def __init__(self, processed_dir: Path):
        """Initialize the attributes."""
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf(self, pdf_path: Path) -> Dict:
        """Extract contents from PDF.

        The content is converted to Markdown, in order to preserve
        more of the content and to make chunking more efficient.

        This method assumes that the markdown version of the extracts
        from the source PDF document will then be chunked using the
        langchain.text_splitter.MarkdownTextSplitter class. If the
        original document is well-described by the use of Markdown,
        the chunking will hew more closely to the inherent structure
        of the source document, instead of simply chunking based on
        counts of characters and ignoring any structure.

        https://pymupdf.readthedocs.io/en/latest/rag.html#rag-outputting-as-md
        """
        print(
            f'extract_pdf: {pdf_path.name}'
        )
        try:
            # TODO: Figure out how to handle headers and footers
            markdown_text = pymupdf4llm.to_markdown(
                pdf_path
            )

        except Exception as e:
            print(
                f'PDF extraction excepted. {e}'
            )
            return None

        return {
            'source': pdf_path.name,
            'type': 'pdf',
            'text': markdown_text,
            'metadata': {
                'source': pdf_path.name,
                'format': 'markdown'
            }
        }

    def extract_txt(self, txt_path: Path) -> Dict:
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

                if body:
                    # TODO: Add all the recipients
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

    def process_all(self, raw_dir: Path) -> int:
        """Process the collection of documents.

        File types are determine solely by the extension. The
        method can handle: pdf, txt, docx, odt, and mbox.
        Each of the document types are expected to be in a sub-directory
        of their name. For example, docx files are in the docx
        sub-directory.

        All of the documents are processed together. The
        extracted text is written to a single file, which
        is terribly resource intensive.

        Returns the quantity of documents processed.
        """
        all_docs = []

        for file_type in ['pdf', 'txt', 'docx', 'odt', 'mbox']:
            print(f'Processing {file_type}')
            for file_path in (raw_dir / file_type).glob(f'*.{file_type}'):
                doc = getattr(self, f'extract_{file_type}')(file_path)
                if doc:
                    all_docs.append(doc)

        output_file = self.processed_dir / DOCUMENTS_FILE
        with open(output_file, 'w') as f:
            for doc in all_docs:
                f.write(dumps(doc) + '\n')

        print(f'Processed {len(all_docs)} documents -> {output_file}')
        return len(all_docs)
