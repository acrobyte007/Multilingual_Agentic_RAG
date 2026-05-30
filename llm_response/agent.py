from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
from retrieval.semantic_search import retrieve
from dotenv import load_dotenv
load_dotenv()

mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=1,
)

async def get_response(query: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
        
Guidelines:
- Answer only using the information from the context
- Be concise and accurate
- If the context doesn't contain the answer, say "I cannot find this information in the provided documents"
- Do not make up or assume information
- Cite the relevant parts of context when possible

Context:
{context}"""),
        ("human", "{query}")
    ])
    
    chain = prompt | mistral_primary
    
    response = await chain.ainvoke({
        "context": context,
        "query": query
    })
    
    return response.content