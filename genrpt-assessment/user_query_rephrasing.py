import os

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REPHRASE_PROMPT_TEXT = """
You are a helpful assistant. Rephrase the user's question clearly and concisely,
while preserving its original meaning and intent.
"""


class RephrasedQueryOutput(BaseModel):
    """
    Schema representing the output of a rephrased user query.

    Attributes:
        rephrased_query (str): The reworded version of the original user query,
        maintaining the same intent and meaning.
    """
    rephrased_query: str


def generate_rephrased_query(original_question: str) -> str:
    """
    Uses an LLM to rephrase the given user question for clarity or downstream processing.

    Args:
        original_question (str): The raw user input to be rephrased.

    Returns:
        str: A rephrased version of the input question with the same intent.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPHRASE_PROMPT_TEXT),
            ("human", "User question: {original_question}"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o"
    )

    rephrasing_chain = prompt | llm.with_structured_output(
        RephrasedQueryOutput)

    response = rephrasing_chain.invoke(
        {"original_question": original_question})

    return response.rephrased_query


if __name__ == "__main__":
    query = "What are the matches where Gujarat faced Mumbai?"
    print("\nGenerated Rephrased Query:\n", generate_rephrased_query(query))
