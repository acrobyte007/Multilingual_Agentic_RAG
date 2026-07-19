import time
from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool,ToolRuntime
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain.agents.middleware import PIIMiddleware
from features.retrieval.pipe_line import top_k_retrieval
from logger.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=1,
)

@dataclass
class UserContext:
    namespace: str
    doc_ids: List[str]

@tool
async def search_and_respond(runtime: ToolRuntime[UserContext],query: str,translated_queries: Dict[str, str]) -> str:
    """The search_and_respond tool takes the following input:
    "query": "string",
    "translated_queries": {"en": "english query", "hi": "hindi query", "bang": "bengali query"}"""
    context = runtime.context
    namespace = context.namespace
    doc_ids = context.doc_ids
    logger.info(f"Searching documents: namespace={namespace}, query={query}, doc_ids={doc_ids}")
    logger.info(f"Translated queries: {translated_queries}")
    chunks = await top_k_retrieval(namespace, query, doc_ids, translated_queries)
    return chunks

tools = [search_and_respond]


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

"""
class RAGAgent(BaseModel):
    answer: str =Field(description="The answer to the question")


agent = create_agent(
        mistral_primary,
        tools,
        context_schema=UserContext,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            PIIMiddleware( "email",strategy="redact",apply_to_input=True,),
            PIIMiddleware("credit_card",strategy="mask",apply_to_input=True,),
            PIIMiddleware("api_key",detector=r"sk-[a-zA-Z0-9]{32}",strategy="block",apply_to_input=True,),
        ],
        response_format=RAGAgent)

async def get_rag_answer(
    namespace: str,
    query: str,
    doc_ids: List[str],
    conversation: List = None
) -> str:
    
    

    user_payload = f"""
user_query: {query}
conversation_history: {conversation if conversation else "No previous conversation"}
"""

    time_1 = time.time()
    result = await agent.ainvoke(
        {"messages": [ {"role": "user", "content": user_payload}]},
        context=UserContext(namespace=namespace, doc_ids=doc_ids)
        )
    response = result["structured_response"]
    time_2 = time.time()
    logger.info(f"Time taken for RAG agent to respond: {time_2 - time_1} seconds")
    return response.answer