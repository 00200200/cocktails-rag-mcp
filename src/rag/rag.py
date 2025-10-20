from os import environ
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.db.db_handler import DBHandler


class State(TypedDict):
    user_question: str
    context: List[Document]
    reranked_context: List[Document]
    response: str


class RAG:
    """RAG pipeline implementation"""

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        """Initialize the RAG pipeline components."""
        load_dotenv()
        self.llm = ChatGroq(model=model_name, api_key=environ.get("GROQ_API_KEY"))
        self.db = DBHandler()
        self.reranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)

    def retrieve(self, state: State, k: int = 10) -> State:
        """Retrieve relevant documents using vector similarity search."""
        state["context"] = self.db.search(state["user_question"], k=k)
        return state

    def rerank_context(self, state: State, k: int = 3) -> State:
        """Rerank retrieved documents using cross-encoder for better relevance."""
        pairs = [(state["user_question"], doc.page_content) for doc in state["context"]]
        scores = self.reranker_model.score(pairs)
        sorted_indices = scores.argsort()[::-1][:k]
        state["reranked_context"] = [state["context"][i] for i in sorted_indices]
        return state

    def generate_answer(self, state: State) -> State:
        """Generate answer using the LLM."""
        context = "\n\n".join([doc.page_content for doc in state["reranked_context"]])
        question = state["user_question"]
        messages = [
            SystemMessage(
                content="""You are a cocktail database assistant that answers ONLY based on provided sources.
                If in context is no cocktail that matches the question, say so clearly.
                """
            ),
            HumanMessage(content=f"Question: {question}\n\n{context}"),
        ]
        response = self.llm.invoke(messages)
        state["response"] = response.content
        return state

    @staticmethod
    def debug_print(state: State) -> None:
        """Debug print the state of the RAG pipeline."""
        print(f"\nQUESTION: {state['user_question']}")

        print(f"\nRETRIEVED: {len(state['context'])} docs")
        for doc in state["context"]:
            print(doc.page_content)

        print(f"RERANKED: {len(state['reranked_context'])} docs")

        for doc in state["reranked_context"]:
            print(doc.page_content)

        print(f"\nRESPONSE:\n{state['response']}\n")


if __name__ == "__main__":
    rag = RAG()
    input_question = input("Enter your question: ")
    state = {
        "user_question": input_question,
        "context": [],
        "reranked_context": [],
        "response": "",
    }
    state = rag.retrieve(state)
    state = rag.rerank_context(state)
    state = rag.generate_answer(state)
    debug_mode = input("Debug mode? (y/n): ")
    if debug_mode.lower() == "y":
        rag.debug_print(state)
    else:
        print(state["response"])
