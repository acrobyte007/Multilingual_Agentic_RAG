from logger.logger import get_logger
logger = get_logger(__name__)
import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from features.retrieval.pipe_line import top_k_retrieval
from dotenv import load_dotenv
load_dotenv()


@tool
async def search_and_respond(namespace: str, query: str, doc_ids: List[str], translated_queries: Dict[str, str]) -> str:
    """Search documents and return relevant answer based on the query and document IDs.
    Use translated_queries to search in multiple languages every time."""
    logger.info(f"Searching documents: namespace={namespace}, query={query}, doc_ids={doc_ids}")
    logger.info(f"Translated queries: {translated_queries}")
    chunks = await top_k_retrieval(namespace, query, doc_ids, translated_queries)
    return chunks

mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=1,
)

class RAGAgent(BaseModel):
    answer: str =Field(description="The answer to the question")

tools = [search_and_respond]
agent = create_agent(mistral_primary, tools,response_format=RAGAgent)

async def get_rag_answer(
    namespace: str,
    query: str,
    doc_ids: List[str],
    translated_queries: Dict[str, str] = None,
    conversation: List = None
) -> str:
    
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
LANGUAGE HANDLING
• The original user query may be in English, Hindi, or Bengali
• translated_queries dictionary contains translations of the query in different languages
• If the user query is in Hindi, respond in Hindi using Devanagari script
• If the user query is in Bengali, respond in Bengali using Bengali script
• If the user query is in English, respond in English
• Maintain consistent language throughout your response

TOOL INPUT
The search_and_respond tool takes the following input:
    "namespace": "string",
    "query": "string",
    "doc_ids": ["string"],
    "translated_queries": {"en": "english query", "hi": "hindi query", "bang": "bengali query"}
"""

    user_payload = f"""
namespace: {namespace}
user_query: {query}
document_ids: {doc_ids}
translated_queries: {translated_queries if translated_queries else "No translations available"}
conversation_history: {conversation if conversation else "No previous conversation"}
"""

    time_1 = time.time()
    result = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload}
        ]
    })
    response = result["structured_response"]
    time_2 = time.time()
    logger.info(f"Time taken for RAG agent to respond: {time_2 - time_1} seconds")
    return response.answer