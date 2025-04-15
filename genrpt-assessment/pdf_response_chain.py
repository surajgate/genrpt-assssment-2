import os
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
NUM_DOCS_TO_RANK = int(os.getenv("NUM_CHUNKS_TO_HYBRID_SEARCH", default="30"))
NUM_TOP_DOCS = int(os.getenv("NUM_CHUNKS_RETRIEVED", default="30"))
DENSE_WEIGHT = float(os.getenv("DENSE_VECTOR_WEIGHT", default="0.65"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_VECTOR_WEIGHT", default="0.35"))

QA_PROMPT_TEMPLATE = """
You are a factual QA assistant for IPL cricket matches. Follow these guidelines:
1. Provide answers based **only on the context** provided.
2. **Do not make assumptions** or provide information beyond the context.
3. If the answer is **uncertain** or not available in the context, respond with: "Sorry, I don't know about this."
4. **Structure your response clearly** and provide only the necessary information.

User Question: {input}

Context:
{context}
"""


class AnswerWithConfidenceSchema(BaseModel):
    """
    Schema to validate and structure the output from the language model.

    Attributes:
        answer (str): Final answer to the user query, based on provided context.
        confidence_score (float): Confidence score of the generated answer.
    """
    answer: str
    confidence_score: float


class HybridSimilaritySearchRetriever(VectorStoreRetriever):
    """
    Custom hybrid retriever using dense and sparse similarity scoring.

    Assumes the vector store supports both `similarity_search_with_score` (sync)
    and `asimilarity_search_with_score` (async) interfaces.

    Attributes:
        vectorstore (VectorStore): Backend vector store for document retrieval.
        search_type (str): Retrieval method; defaults to 'similarity_score'.
    """

    vectorstore: VectorStore

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        """Synchronously fetch relevant documents using similarity scores."""
        _kwargs = self.search_kwargs | kwargs
        docs = self.vectorstore.similarity_search_with_score(query, **_kwargs)
        return [doc for doc, _ in docs]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously fetch relevant documents using similarity scores."""
        _kwargs = self.search_kwargs | kwargs
        docs = await self.vectorstore.asimilarity_search_with_score(query, **_kwargs)
        return [doc for doc, _ in docs]


def generate_answer_from_pdf_context(question: str) -> tuple[str, float]:
    """
    Generates an answer to a user question by:
    1. Retrieving relevant context from a Milvus vector store using hybrid search.
    2. Passing the context and question to a GPT-4o model via a structured prompt.

    Args:
        question (str): The user's question in natural language.

    Returns:
        tuple[str, float]: A tuple containing:
            - The final answer string.
            - The confidence score associated with the answer.
    """
    embedding_model = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY, model="text-embedding-3-large"
    )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o"
    )

    vector_store = Milvus(
        connection_args={"uri": MILVUS_CONNECTION_URI},
        embedding_function=embedding_model,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        collection_name=COLLECTION_NAME,
        search_params=[
            {"metric_type": "L2", "params": {"ef": NUM_DOCS_TO_RANK + 100}},
            {"metric_type": "BM25"},
        ],
        consistency_level="Strong",
    )

    prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

    qa_prompt_chain = (
        RunnablePassthrough.assign(context=lambda x: x["context"])
        | prompt
        | llm.with_structured_output(AnswerWithConfidenceSchema)
    )

    retriever_config = {
        "k": NUM_TOP_DOCS,
        "fetch_k": NUM_DOCS_TO_RANK,
        "ranker_type": "weighted",
        "ranker_params": {"weights": [DENSE_WEIGHT, SPARSE_WEIGHT]},
    }

    retriever = HybridSimilaritySearchRetriever(
        vectorstore=vector_store,
        search_kwargs=retriever_config,
    )

    retrieval_chain = (lambda x: x["input"]) | retriever

    hybrid_rag_chain = RunnablePassthrough.assign(context=retrieval_chain).assign(
        answer=qa_prompt_chain
    )

    response = hybrid_rag_chain.invoke({"input": question})["answer"]

    return response.answer, response.confidence_score


if __name__ == "__main__":
    """
    Example usage of the hybrid PDF QA pipeline. This script:
    - Accepts a hardcoded user question.
    - Retrieves relevant document chunks from the Milvus vector store using hybrid search.
    - Uses an LLM to generate a response based on the retrieved context.
    - Prints the final answer and its associated confidence score.
    """
    user_input_question = "What is CSK’s home ground?"

    answer, confidence_score = generate_answer_from_pdf_context(
        user_input_question)

    print("\nQuestion:", user_input_question)
    print("\nLLM Answer:\n", answer)
    print("\nConfidence Score:", confidence_score)
