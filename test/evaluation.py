from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
    # other params...
)



class ChunkRelevanceClassification(BaseModel):
    chunk_index: int = Field(
        description="The 0-based index or position of the retrieved chunk in the context list."
    )
    reason: str = Field(
        description="Brief explanation justifying why this chunk is relevant or irrelevant to answering the query."
    )
    is_relevant: int = Field(
        description="Binary relevance score: 1 if the chunk contains useful/relevant information, 0 if irrelevant.",
        ge=0,
        le=1
    )

class ContextPrecisionResponse(BaseModel):
    classifications: List[ChunkRelevanceClassification] = Field(
        description="List of relevance classifications ordered by chunk position."
    )


class ContextRecallClassification(BaseModel):
    statement: str = Field(
        description="A distinct factual claim or sentence extracted from the ground truth answer."
    )
    reason: str = Field(
        description="Explanation detailing whether the statement can be directly attributed to the retrieved context."
    )
    attributed: int = Field(
        description="Binary indicator: 1 if the statement is present/supported in the retrieved context, 0 otherwise."
    )

class ContextRecallResponse(BaseModel):
    classifications: List[ContextRecallClassification] = Field(
        description="A list of classifications for every statement in the ground truth."
    )



class StatementVerification(BaseModel):
    claim: str = Field(
        description="An atomic factual claim extracted from the generated answer."
    )
    reason: str = Field(
        description="Reasoning explaining whether the claim is supported by the retrieved context."
    )
    verdict: int = Field(
        description="1 if the claim is supported by the context, 0 if unsupported/hallucinated.",
        ge=0,
        le=1
    )

class FaithfulnessResponse(BaseModel):
    claims: List[StatementVerification] = Field(
        description="List of all extracted claims and their verification verdicts."
    )

def calculate_context_recall(llm_output: dict) -> float:
    statements = llm_output.get("statements", [])
    if not statements:
        return 0.0
    attributed_count = sum(1 for item in statements if item.get("attributed") == 1)
    total_statements = len(statements)
    return attributed_count / total_statements

def calculate_context_precision(relevance_list: list[int]) -> float:
    total_relevant = sum(relevance_list)
    if total_relevant == 0:
        return 0.0
    
    running_relevant_count = 0
    sum_precision = 0.0
    
    for position, v_k in enumerate(relevance_list, start=1):
        if v_k == 1:
            running_relevant_count += 1
            precision_at_k = running_relevant_count / position
            sum_precision += precision_at_k
    return sum_precision / total_relevant

def calculate_faithfulness(llm_response: FaithfulnessResponse) -> float:
    claims = llm_response.claims
    if not claims:
        return 0.0
    supported_claims_count = sum(1 for item in claims if item.verdict == 1)
    total_claims_count = len(claims)
    return supported_claims_count / total_claims_count


recall_model=model.with_structured_output(ContextRecallResponse)
precision_model=model.with_structured_output(ContextPrecisionResponse)
faithfulness_model=model.with_structured_output(FaithfulnessResponse)