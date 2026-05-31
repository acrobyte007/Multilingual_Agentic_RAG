from logger.logger import get_logger
logger = get_logger(__name__)
import time
from typing import List, Optional
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from retrieval.pipe_line import top_k_retrieval
from dotenv import load_dotenv
load_dotenv()


@tool
async def search_and_respond(namespace: str, query: str, doc_ids: List[str]) -> str:
    """Search documents and return relevant answer based on the query and document IDs"""
    logger.info(f"Searching documents: namespace={namespace}, query={query}, doc_ids={doc_ids}")
    chunks = await top_k_retrieval(namespace, query, doc_ids)
    return chunks

mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=1,
)

tools = [search_and_respond]
agent = create_agent(mistral_primary, tools)

async def get_rag_answer(
    namespace: str,
    query: str,
    doc_ids: List[str],
    conversation: dict = None
) -> str:
    
    SYSTEM_PROMPT = """
You are a knowledgeable assistant that answers questions based on provided documents.

TONE & STYLE
• Be friendly, polite, and professional
• Sound natural and human
• Keep responses simple and easy to understand

IMPORTANT RULES
• Use search_and_respond tool to find answers from documents
• Answer must be based solely on retrieved document chunks
• If no relevant information found, state clearly that information is not available
• Always cite sources by mentioning document IDs

RESPONSE GUIDELINES
• Information Found → Provide answer with sources
• No Information Found → State information not found
• Use markdown formatting with "-" for steps or bullet points when needed
TOOL INPUT
The search_and_respond tool takes the following input:
    "namespace": "string",
    "query": "string",
    "doc_ids": ["string"]
"""

    user_payload = f"""
namespace: {namespace}
user_query: {query}
document_ids: {doc_ids}
conversation_history: {conversation if conversation else "No previous conversation"}
"""

    time_1 = time.time()

    result = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload}
        ]
    })

    time_2 = time.time()

    logger.info(f"Time taken for RAG agent to respond: {time_2 - time_1} seconds")

    return result["messages"][-1].content