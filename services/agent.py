import time
from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool,ToolRuntime
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware,ModelCallLimitMiddleware,ToolCallLimitMiddleware
from langchain.agents.middleware import ToolRetryMiddleware,ModelRetryMiddleware
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from features.retrieval.pipe_line import top_k_retrieval
from logger.logger import get_logger
from dotenv import load_dotenv
import json

load_dotenv()
logger = get_logger(__name__)


mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0.7,
    max_retries=2,
    timeout=60,
)

@dataclass
class UserContext:
    namespace: str
    doc_ids: List[str]

class RetryableToolError(RuntimeError):
    """Temporary error. Agent may retry."""

class NonRetryableToolError(ValueError):
    """Bad input. Agent should not retry."""

class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass

class ConnectionError(Exception):
    """Custom exception for connection errors."""
    pass

def should_retry(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if hasattr(error, "status_code"):
        return error.status_code in (429, 503)
    return False

@tool
async def search(
    runtime: ToolRuntime[UserContext],
    query: str,
    translated_queries: Dict[str, str],
):
    """
    Search the knowledge base.

    Args:
        query: User's original query.
        translated_queries: Dictionary containing translated queries.
            Example:
            {
                "en": "...",
                "hi": "...",
                "bn": "..."
            }

    Returns:
        List of retrieved chunks.
    """

    if runtime is None:
        raise NonRetryableToolError("runtime cannot be None.")

    if not isinstance(query, str) or not query.strip():
        raise NonRetryableToolError("query must be a non-empty string.")

    if not isinstance(translated_queries, dict):
        raise NonRetryableToolError("translated_queries must be a dictionary.")

    required_languages = {"en", "hi", "bn"}

    missing = required_languages - translated_queries.keys()
    if missing:
        raise NonRetryableToolError(
            f"translated_queries is missing required languages: {sorted(missing)}"
        )

    for lang in required_languages:
        value = translated_queries.get(lang)

        if not isinstance(value, str):
            raise NonRetryableToolError(
                f"translated_queries['{lang}'] must be a string."
            )

        if not value.strip():
            raise NonRetryableToolError(
                f"translated_queries['{lang}'] cannot be empty."
            )

    context = runtime.context

    if context is None:
        raise RetryableToolError("Runtime context is missing.")

    if not context.namespace:
        raise NonRetryableToolError("namespace is missing.")

    if not context.doc_ids:
        raise NonRetryableToolError("doc_ids is missing.")

    try:
        chunks = await top_k_retrieval(
            context.namespace,
            query,
            context.doc_ids,
            translated_queries,
        )

        if chunks is None:
            raise RetryableToolError("Retriever returned None.")

        logger.info(
            "Retrieved %d chunks | namespace=%s | doc_ids=%s",
            len(chunks),
            context.namespace,
            context.doc_ids,
        )

        return chunks

    except TimeoutError as e:
        logger.exception("Retrieval timed out.")
        raise RetryableToolError(
            "Temporary retrieval timeout. Please retry."
        ) from e

    except ConnectionError as e:
        logger.exception("Retriever connection failed.")
        raise RetryableToolError(
            "Temporary retrieval connection failure. Please retry."
        ) from e

    except Exception as e:
        logger.exception("Unexpected error during retrieval.")
        raise RetryableToolError(
            f"Search tool failed: {type(e).__name__}: {e}"
        ) from e

tools = [search]


SYSTEM_PROMPT = """
You are a knowledgeable assistant that answers questions based on provided documents.
TONE & STYLE
• Be friendly, polite, and professional
• Sound natural and human
• Keep responses simple and easy to understand
GREETING & CLOSING
• Start with a greeting like "Hello! How can I assist you today?"
IMPORTANT RULES
• Use search_and_respond tool to find answers from documents
• Answer must be based solely on retrieved document chunks
• If no relevant information found, state clearly that information is not available

RESPONSE GUIDELINES
• Information Found → Provide answer with sources
• No Information Found → State information not found
• Use markdown formatting with "-" for steps or bullet points when needed
• Respond in the SAME language as the user's original query,not the language of the retrieved documents
• If the answer can be given in short form, provide a concise response
LANGUAGE HANDLING
• The original user query may be in English, Hindi, or Bengali
• translated_queries dictionary contains translations of the query in different languages
• If the user query is in Hindi, respond in Hindi using Devanagari script
• If the user query is in Bengali, respond in Bengali using Bengali script
• If the user query is in English, respond in English
• Maintain consistent language throughout your response
"""
class RAGAgent(BaseModel):
    answer: str =Field(description="The answer to the question")


agent = create_agent(
        mistral_primary,
        tools,
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            PIIMiddleware( "email",strategy="redact",apply_to_input=True,),
            PIIMiddleware("credit_card",strategy="mask",apply_to_input=True,),
            PIIMiddleware("api_key",detector=r"sk-[a-zA-Z0-9]{32}",strategy="block",apply_to_input=True,),
            SummarizationMiddleware(model=mistral_primary,trigger=("tokens", 4000)),
            ModelCallLimitMiddleware(thread_limit=20,run_limit=3,exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search",thread_limit=20,run_limit=2),
            ToolRetryMiddleware(max_retries=2,backoff_factor=2.0,initial_delay=1.0,max_delay=60.0,jitter=True,tools=["search"],retry_on=(ConnectionError, TimeoutError),on_failure="continue"),
            ModelRetryMiddleware(max_retries=3,retry_on=(TimeoutError, ConnectionError,should_retry),backoff_factor=1.5,on_failure="continue")
        ],
        response_format=RAGAgent)
@traceable(run_type="llm")
async def get_rag_answer(
    namespace: str,
    query: str,
    doc_ids: List[str],
    conversation_id: str = None,
) -> tuple:
    logger.info(f"Getting RAG answer for query: {query}")
    thread_config = {"configurable": {"thread_id": conversation_id}}
    time_1 = time.time()
    result = await agent.ainvoke(
        {"messages": [ {"role": "user", "content": query}]},
        thread_config,
        context=UserContext(namespace=namespace, doc_ids=doc_ids)
        )
    response = result["structured_response"]
    time_2 = time.time()
    tool_message = next((m for m in result["messages"] if isinstance(m, ToolMessage)),None,)
    sources = None
    if tool_message:
        sources = json.loads(tool_message.content)
    logger.info(f"RAG answer generated in {time_2 - time_1:.2f} seconds")
    return response.answer,sources
