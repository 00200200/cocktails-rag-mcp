from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.data.dataio import load_documents


class DBHandler:
    """DBHandler class for handling the vector database."""

    def __init__(self):
        """Initialize the database handler and load or create the FAISS index."""
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        project_root = Path(__file__).parent.parent.parent
        faiss_path = project_root / "faiss_index"

        if faiss_path.exists():
            self.db = FAISS.load_local(
                str(faiss_path), embeddings, allow_dangerous_deserialization=True
            )
        else:
            documents = load_documents()
            self.db = FAISS.from_documents(documents, embeddings)
            self.db.save_local(str(faiss_path))

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant cocktail documents using semantic similarity.

        Args:
            query: The search query string
            k: Number of most similar documents to return

        Returns:
            List of Document objects most similar to the query
        """
        return self.db.similarity_search(query, k=k)
