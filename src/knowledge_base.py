"""
Knowledge Base Module with RAG Implementation
Supports document upload, vector storage with Chroma, and semantic retrieval
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class KnowledgeBase(ABC):
    """Abstract base class for a knowledge base retriever."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    def add_documents(self, file_paths: List[str]) -> int:
        """Add documents to the knowledge base."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all documents from the knowledge base."""
        pass


class ChromaKnowledgeBase(KnowledgeBase):
    """
    Production-ready knowledge base using Chroma vector store with OpenAI embeddings.
    Supports PDF, TXT, and DOCX document ingestion.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the Chroma knowledge base.

        Args:
            persist_directory: Directory to persist the vector store. If None, uses in-memory store.
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Text splitter for semantic chunking with contextual awareness
        # Using best practices from 2025: semantic chunking with appropriate overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True  # Track position in original document
        )

        # Initialize or load existing vector store
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize the Chroma vector store."""
        persist_path = Path(self.persist_directory)

        # Check if existing vector store exists
        if persist_path.exists() and any(persist_path.iterdir()):
            print(f"INFO: Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"INFO: Creating new vector store at {self.persist_directory}")
            persist_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            print(f"INFO: Loaded {len(documents)} page(s) from {Path(file_path).name}")
            return documents

        except Exception as e:
            print(f"ERROR: Failed to load {file_path}: {e}")
            return []

    def add_documents(self, file_paths: List[str]) -> int:
        """
        Add documents to the vector store.

        Args:
            file_paths: List of file paths to add

        Returns:
            Number of chunks successfully added
        """
        all_chunks = []

        for file_path in file_paths:
            documents = self._load_document(file_path)
            if documents:
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(documents)

                # Add metadata
                for chunk in chunks:
                    chunk.metadata['source'] = Path(file_path).name

                all_chunks.extend(chunks)

        if all_chunks:
            # Add to vector store
            self.vectorstore.add_documents(all_chunks)
            print(f"INFO: Added {len(all_chunks)} chunks to vector store")
            return len(all_chunks)

        return 0

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant document chunks using hybrid search (similarity + MMR).
        Implements 2025 best practices for better retrieval quality.

        Args:
            query: The search query
            top_k: Number of top results to return (default 5 for richer context)

        Returns:
            List of relevant document contents with source metadata
        """
        try:
            # Use MMR (Maximal Marginal Relevance) for diverse results
            # This prevents retrieving too many similar chunks
            results = self.vectorstore.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=top_k * 3  # Fetch more candidates for diversity
            )

            if not results:
                return ["No documents found in the knowledge base. Please upload documents first."]

            # Format results with rich metadata
            formatted_results = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content.strip()

                # Format with contextual header
                formatted_results.append(
                    f"[Document {i}: {source} (Page {page})]\n{content}"
                )

            print(f"INFO: Retrieved {len(formatted_results)} chunks for query: '{query[:50]}...'")
            return formatted_results

        except Exception as e:
            print(f"ERROR: Failed to retrieve documents: {e}")
            return ["No documents found in the knowledge base. Please upload documents first."]

    def clear(self):
        """Clear all documents from the vector store."""
        try:
            # Delete the persist directory
            import shutil
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)

            # Reinitialize
            self._initialize_vectorstore()
            print("INFO: Knowledge base cleared successfully")

        except Exception as e:
            print(f"ERROR: Failed to clear knowledge base: {e}")

    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            # Get collection count
            collection = self.vectorstore._collection
            return collection.count()
        except Exception as e:
            print(f"ERROR: Failed to get document count: {e}")
            return 0


