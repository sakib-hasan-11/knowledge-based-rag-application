"""
Document Parser and Ingestion Module (Phase 2.4, 2.6)

Parses HTML documents, extracts specific sections, and creates LangChain Document objects
with proper metadata and error handling for production use.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config
from .document_loader import S3DocumentLoader
from .logging_config import create_logger

# Target sections to extract from 10-K/10-Q documents
TARGET_SECTIONS = {
    "Item 1": {"full_name": "Business", 
               "alias": ["item 1", "business"], 
               "priority": 1},
               
    "Item 1A": {
        "full_name": "Risk Factors",
        "alias": ["item 1a", "risk factors"],
        "priority": 2,
    },
    "Item 7": {
        "full_name": "Management's Discussion and Analysis",
        "alias": ["item 7", "md&a", "management discussion analysis"],
        "priority": 3,
    },
    "Item 8": {
        "full_name": "Financial Statements",
        "alias": ["item 8", "financial statements"],
        "priority": 4,
    },
}


@dataclass
class DocumentMetadata:
    """Metadata for extracted document sections"""

    section: str
    section_full_name: str
    company: str
    fiscal_year: Optional[str] = None
    filing_type: Optional[str] = None
    source_file: Optional[str] = None
    extraction_date: str = ""

    def __post_init__(self):
        if not self.extraction_date:
            self.extraction_date = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class HTMLDocumentParser:
    """
    Parser for extracting specific sections from HTML financial documents.
    Handles SEC filings and other corporate documents with error handling.
    """

    def __init__(
        self,
        target_sections: Optional[Dict] = None,
        logger_name: str = "HTMLDocumentParser",
    ):
        """
        Initialize HTML document parser.

        Args:
            target_sections: Dictionary of sections to extract
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)
        self.target_sections = target_sections or TARGET_SECTIONS
        self.logger.info(
            f"Initialized HTMLDocumentParser",
            {"num_target_sections": len(self.target_sections)},
        )

    def parse_html_content(self, html_content: str) -> Optional[BeautifulSoup]:
        """
        Parse HTML content using BeautifulSoup with error handling.

        Args:
            html_content: Raw HTML string

        Returns:
            BeautifulSoup object or None if parsing fails
        """
        try:
            soup = BeautifulSoup(html_content, "html5lib")
            self.logger.debug(f"HTML parsed successfully")
            return soup
        except Exception as e:
            self.logger.error(
                f"Error parsing HTML content: {str(e)}",
                {"exception_type": type(e).__name__, "html_length": len(html_content)},
            )
            return None

    def extract_section(self, soup: BeautifulSoup, section_key: str) -> Dict[str, Any]:
        """
        Extract a specific section from parsed HTML document.

        Args:
            soup: BeautifulSoup parsed document
            section_key: Key from TARGET_SECTIONS (e.g., 'Item 1')

        Returns:
            Dict with 'found', 'text', 'start_idx', 'end_idx'
        """
        try:
            section_info = self.target_sections.get(section_key)
            if not section_info:
                return {"found": False, "text": None}

            # Search patterns for headers
            patterns = [section_key.lower()] + section_info["alias"]
            text_content = soup.get_text()

            # Find section header
            start_idx = -1
            for pattern in patterns:
                idx = text_content.lower().find(pattern)
                if idx != -1:
                    start_idx = idx
                    break

            if start_idx == -1:
                self.logger.debug(f"Section {section_key} not found in document")
                return {"found": False, "text": None}

            # Find next section header or end of document
            remaining_text = text_content[start_idx:]
            item_pattern = r"\nItem \d+"
            next_match = re.search(item_pattern, remaining_text)

            if next_match:
                end_idx = start_idx + next_match.start()
            else:
                end_idx = len(text_content)

            section_text = text_content[start_idx:end_idx].strip()

            return {
                "found": True,
                "text": section_text,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": len(section_text),
            }

        except Exception as e:
            self.logger.error(
                f"Error extracting section {section_key}: {str(e)}",
                {"section_key": section_key, "exception_type": type(e).__name__},
            )
            return {"found": False, "text": None, "error": str(e)}

    def extract_all_sections(self, html_content: str) -> Dict[str, Dict]:
        """
        Extract all target sections from HTML document.

        Args:
            html_content: Raw HTML string

        Returns:
            Dict mapping section keys to extracted content
        """
        try:
            soup = self.parse_html_content(html_content)
            if not soup:
                return {}

            extracted = {}

            # Sort sections by priority
            sorted_sections = sorted(
                self.target_sections.items(), key=lambda x: x[1]["priority"]
            )

            for section_key, section_info in sorted_sections:
                result = self.extract_section(soup, section_key)
                extracted[section_key] = result

            found_count = sum(1 for r in extracted.values() if r.get("found"))
            self.logger.info(
                f"Extracted sections from document",
                {
                    "total_sections_targeted": len(self.target_sections),
                    "sections_found": found_count,
                },
            )
            return extracted

        except Exception as e:
            self.logger.error(
                f"Error extracting all sections: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return {}

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and artifacts.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)
            # Remove special characters that might be encoding artifacts
            text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
            return text.strip()
        except Exception as e:
            self.logger.error(
                f"Error cleaning text: {str(e)}",
                {"exception_type": type(e).__name__, "text_length": len(text)},
            )
            return text.strip()


class DocumentIngestionPipeline:
    """
    End-to-end pipeline for loading, parsing, extracting sections,
    and enriching documents with metadata.
    """

    def __init__(
        self,
        s3_loader: Optional[S3DocumentLoader] = None,
        html_parser: Optional[HTMLDocumentParser] = None,
        logger_name: str = "DocumentIngestionPipeline",
    ):
        """
        Initialize the pipeline with loader and parser.

        Args:
            s3_loader: S3DocumentLoader instance (auto-created if None)
            html_parser: HTMLDocumentParser instance (auto-created if None)
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)
        self.s3_loader = s3_loader or S3DocumentLoader()
        self.html_parser = html_parser or HTMLDocumentParser()
        self.extracted_documents: List[Document] = []

        self.logger.info("DocumentIngestionPipeline initialized")

    def process_document(
        self, document_source: str, company: Optional[str] = None, from_s3: bool = False
    ) -> List[Document]:
        """
        Process a single document and extract target sections.

        Args:
            document_source: S3 key or local file path
            company: Company name override
            from_s3: Whether to load from S3 (True) or local (False)

        Returns:
            List of LangChain Document objects with metadata
        """
        try:
            # Load document content
            if from_s3:
                html_content = self.s3_loader.load_document(document_source)
                file_metadata = self.s3_loader.get_file_metadata(document_source)
            else:
                html_content = self.s3_loader.load_local_html(document_source)
                file_metadata = {
                    "source_file": document_source,
                    "company": company or "UNKNOWN",
                }

            if not html_content:
                self.logger.error(f"Failed to load document: {document_source}")
                return []

            # Extract sections
            extracted_sections = self.html_parser.extract_all_sections(html_content)

            # Create LangChain Documents with metadata
            langchain_docs = []

            for section_key, section_data in extracted_sections.items():
                if not section_data.get("found"):
                    continue

                try:
                    # Prepare metadata
                    section_info = TARGET_SECTIONS.get(section_key)
                    doc_metadata = DocumentMetadata(
                        section=section_key,
                        section_full_name=section_info["full_name"],
                        company=file_metadata.get("company", "UNKNOWN"),
                        fiscal_year=file_metadata.get("fiscal_year"),
                        filing_type=file_metadata.get("filing_type"),
                        source_file=file_metadata.get("source_file"),
                    )

                    # Clean text
                    cleaned_text = self.html_parser.clean_text(section_data["text"])

                    # Create LangChain Document
                    doc = Document(
                        page_content=cleaned_text, metadata=doc_metadata.to_dict()
                    )

                    langchain_docs.append(doc)

                except Exception as e:
                    self.logger.error(
                        f"Error processing section {section_key}: {str(e)}",
                        {
                            "section_key": section_key,
                            "exception_type": type(e).__name__,
                        },
                    )
                    continue

            self.extracted_documents.extend(langchain_docs)

            self.logger.info(
                f"Successfully processed document",
                {
                    "document_source": document_source,
                    "sections_extracted": len(langchain_docs),
                },
            )
            return langchain_docs

        except Exception as e:
            self.logger.error(
                f"Error in process_document: {str(e)}",
                {
                    "document_source": document_source,
                    "exception_type": type(e).__name__,
                },
            )
            return []

    def process_batch(
        self, document_sources: List[str], from_s3: bool = False
    ) -> Dict[str, List[Document]]:
        """
        Process multiple documents in batch.

        Args:
            document_sources: List of S3 keys or local file paths
            from_s3: Whether to load from S3 or local

        Returns:
            Dict mapping document source to extracted documents
        """
        results = {}

        for i, source in enumerate(document_sources):
            try:
                self.logger.info(
                    f"Processing batch document {i + 1}/{len(document_sources)}",
                    {"source": source},
                )
                docs = self.process_document(source, from_s3=from_s3)
                results[source] = docs
            except Exception as e:
                self.logger.error(
                    f"Error processing batch document: {str(e)}",
                    {"source": source, "exception_type": type(e).__name__},
                )
                results[source] = []

        self.logger.info(
            f"Batch processing completed",
            {
                "total_documents": len(document_sources),
                "successful": sum(1 for v in results.values() if v),
                "total_documents_extracted": len(self.extracted_documents),
            },
        )
        return results

    def get_all_extracted_documents(self) -> List[Document]:
        """Get all extracted documents from pipeline"""
        return self.extracted_documents

    def save_extracted_documents(self, output_path: str, format: str = "json") -> bool:
        """
        Save extracted documents to file with error handling.

        Args:
            output_path: Path where to save documents
            format: 'json' or 'jsonl'

        Returns:
            True if successful, False otherwise
        """
        try:
            output_data = []

            for doc in self.extracted_documents:
                output_data.append(
                    {"content": doc.page_content, "metadata": doc.metadata}
                )

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            elif format == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in output_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            self.logger.info(
                f"Saved extracted documents",
                {
                    "output_path": output_path,
                    "format": format,
                    "num_documents": len(output_data),
                },
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error saving extracted documents: {str(e)}",
                {
                    "output_path": output_path,
                    "format": format,
                    "exception_type": type(e).__name__,
                },
            )
            return False
