import os
from typing import Dict, Union
from pathlib import Path
import fitz
from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentExtractor:
    def __init__(self):
        pass

    def extract_documents(self, path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
        path = Path(path)
        extracted_docs = {}

        if path.is_file():
            if path.suffix.lower() == ".pdf":
                logger.info(f"Processing file: {path}")
                text = self._extract_pdf_text(path)
                if text:
                    extracted_docs["01"] = {
                        "document": path.stem,
                        "texts": text
                    }
                    logger.info(f"Text extracted from: {path.name}")
                else:
                    logger.warning(f"No text extracted from: {path.name}")
            else:
                logger.error(f"File is not a PDF: {path}")
                raise ValueError(f"File must be a PDF, got: {path.suffix}")

        elif path.is_dir():
            logger.info(f"Processing directory: {path}")
            pdf_files = list(path.glob("*.pdf"))

            if not pdf_files:
                logger.warning(f"No PDF files found in: {path}")
                return extracted_docs

            logger.info(f"Found {len(pdf_files)} PDF files")

            for i, pdf_file in enumerate(sorted(pdf_files), 1):
                try:
                    logger.debug(f"Extracting text ({i}/{len(pdf_files)}): {pdf_file.name}")
                    text = self._extract_pdf_text(pdf_file)

                    if text:
                        doc_id = f"{i:02d}"
                        extracted_docs[doc_id] = {
                            "document": pdf_file.stem,
                            "texts": text
                        }
                        logger.debug(f"Text extracted from: {pdf_file.name}")
                    else:
                        logger.warning(f"No text extracted from: {pdf_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                    continue
        else:
            logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"Path does not exist: {path}")

        logger.info(f"Extraction completed: {len(extracted_docs)} documents processed")
        return extracted_docs

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            doc = fitz.open(pdf_path)
            all_text = []

            logger.debug(f"Extracting {doc.page_count} pages from {pdf_path.name}")

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    all_text.append(text)

            doc.close()

            if all_text:
                full_text = "\n\n".join(all_text)
                logger.debug(f"Extracted {len(full_text)} characters from {pdf_path.name}")
                return full_text
            else:
                logger.warning(f"No text content found in {pdf_path.name}")
                return ""

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
            raise