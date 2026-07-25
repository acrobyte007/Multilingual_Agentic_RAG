import asyncio
import json
import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from database.vector_database import pinecone_service
from services.agent import get_rag_answer

load_dotenv()

pinecone_service.initialize()  # Ensure Pinecone is initialized before any evaluation

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
        le=1,
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
        description="Binary indicator: 1 if the statement is present/supported in the retrieved context, 0 otherwise.",
        ge=0,
        le=1,
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
        le=1,
    )


class FaithfulnessResponse(BaseModel):
    claims: List[StatementVerification] = Field(
        description="List of all extracted claims and their verification verdicts."
    )


# =====================================================================
# 2. SYSTEM PROMPTS & EVALUATOR LLM SETUP
# =====================================================================

PRECISION_SYSTEM_PROMPT = """You are an expert search and retrieval evaluator.
Your task is to evaluate the relevance of each retrieved context chunk with respect to answering the user query.

Instructions:
1. Examine each chunk provided in the context list in order.
2. For each chunk, determine if it contains information directly useful for answering the user's query or ground truth.
3. Assign `is_relevant = 1` if it is helpful, otherwise `0`.
4. Keep track of the 0-based chunk index.
"""

RECALL_SYSTEM_PROMPT = """You are a factual accuracy evaluator for RAG systems.
Your task is to measure Context Recall by checking if all facts from the Ground Truth are present in the Retrieved Context.

Instructions:
1. Decompose the provided `ground_truth` answer into individual atomic factual statements.
2. For each extracted statement, verify whether it can be inferred or supported by the `retrieved_context`.
3. Set `attributed = 1` if supported, otherwise `0`. Provide brief reasoning.
"""

FAITHFULNESS_SYSTEM_PROMPT = """You are an AI hallucination detection judge.
Your task is to evaluate the Faithfulness of an AI-generated answer against the provided retrieved context.

Instructions:
1. Extract every individual factual claim made in the `generated_answer`.
2. Check each claim against the `retrieved_context`.
3. Set `verdict = 1` ONLY if the claim is directly supported by the context. If the claim is unsupported, contradicted, or hallucinated, set `verdict = 0`.
"""

# Base Gemini Evaluator Model
base_eval_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Metric-bound structured models
precision_model = base_eval_model.with_structured_output(
    ContextPrecisionResponse
)
recall_model = base_eval_model.with_structured_output(ContextRecallResponse)
faithfulness_model = base_eval_model.with_structured_output(
    FaithfulnessResponse
)


# =====================================================================
# 3. MATHEMATICAL CALCULATION FUNCTIONS
# =====================================================================


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
    return round(sum_precision / total_relevant, 4)


def calculate_context_recall(llm_output: ContextRecallResponse) -> float:
    statements = llm_output.classifications
    if not statements:
        return 0.0
    attributed_count = sum(1 for item in statements if item.attributed == 1)
    return round(attributed_count / len(statements), 4)


def calculate_faithfulness(llm_response: FaithfulnessResponse) -> float:
    claims = llm_response.claims
    if not claims:
        return 0.0
    supported_claims_count = sum(1 for item in claims if item.verdict == 1)
    return round(supported_claims_count / len(claims), 4)


# =====================================================================
# 4. SINGLE SAMPLE EVALUATION ENGINE
# =====================================================================


def evaluate_single_sample(
    query: str,
    retrieved_context: List[str],
    generated_answer: str,
    ground_truth: str,
) -> Dict:
    """Runs Precision, Recall, and Faithfulness evaluation on a single sample."""
    formatted_context = "\n".join(
        [f"Chunk {i}: {c}" for i, c in enumerate(retrieved_context)]
    )

    # 1. Precision
    prec_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRECISION_SYSTEM_PROMPT),
            (
                "human",
                "Query: {query}\nGround Truth: {ground_truth}\n\nRetrieved Context:\n{context}",
            ),
        ]
    )
    prec_res: ContextPrecisionResponse = (prec_prompt | precision_model).invoke(
        {
            "query": query,
            "ground_truth": ground_truth,
            "context": formatted_context,
        }
    )
    sorted_classifications = sorted(
        prec_res.classifications, key=lambda x: x.chunk_index
    )
    relevance_seq = [c.is_relevant for c in sorted_classifications]
    precision_score = calculate_context_precision(relevance_seq)

    # 2. Recall
    rec_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RECALL_SYSTEM_PROMPT),
            (
                "human",
                "Ground Truth: {ground_truth}\n\nRetrieved Context:\n{context}",
            ),
        ]
    )
    rec_res: ContextRecallResponse = (rec_prompt | recall_model).invoke(
        {"ground_truth": ground_truth, "context": formatted_context}
    )
    recall_score = calculate_context_recall(rec_res)

    # 3. Faithfulness
    faith_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FAITHFULNESS_SYSTEM_PROMPT),
            (
                "human",
                "Generated Answer: {answer}\n\nRetrieved Context:\n{context}",
            ),
        ]
    )
    faith_res: FaithfulnessResponse = (faith_prompt | faithfulness_model).invoke(
        {"answer": generated_answer, "context": formatted_context}
    )
    faithfulness_score = calculate_faithfulness(faith_res)

    return {
        "query": query,
        "generated_answer": generated_answer,
        "ground_truth": ground_truth,
        "scores": {
            "context_precision": precision_score,
            "context_recall": recall_score,
            "faithfulness": faithfulness_score,
        },
        "details": {
            "precision_classifications": [
                c.model_dump() for c in prec_res.classifications
            ],
            "recall_classifications": [
                c.model_dump() for c in rec_res.classifications
            ],
            "faithfulness_claims": [c.model_dump() for c in faith_res.claims],
        },
    }


# =====================================================================
# 5. BENCHMARK PIPELINE (ITEM-BY-ITEM EVALUATION & INCREMENTAL SAVING)
# =====================================================================


async def run_benchmark(
    ground_truth_file: str,
    output_file: str = "evaluation_output.json",
    namespace: str = "anemia_research",
    doc_ids: List[str] = ["doc_001"],
):
    """Loads ground truth file, invokes the imported get_rag_answer agent,

    evaluates results, and saves them sample-by-sample immediately.
    """
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(
            f"Ground truth dataset '{ground_truth_file}' not found."
        )

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    # Resume capability if partially evaluated
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(
                f"Resuming evaluation: {len(results)} samples already completed."
            )
        except json.JSONDecodeError:
            results = []

    start_index = len(results)

    for i in range(start_index, len(ground_truth_data)):
        item = ground_truth_data[i]
        question = item["question"]
        ground_truth = item["answer"]

        print(
            f"\n------------------------------------------------------------"
        )
        print(
            f"Processing Question [{i + 1}/{len(ground_truth_data)}]: {question}"
        )

        try:
            # Step 1: Get output & retrieved chunks from imported agent
            generated_answer, retrieved_context = await get_rag_answer(
                namespace=namespace,
                query=question,
                doc_ids=doc_ids,
            )

            # Ensure retrieved_context is formatted properly as a list of strings
            if isinstance(retrieved_context, str):
                retrieved_context = [retrieved_context]
            elif not retrieved_context:
                retrieved_context = [
                    "No context chunks were retrieved by the agent."
                ]

            print(f"-> Answer Generated: {generated_answer[:80]}...")
            print(f"-> Chunks Retrieved: {len(retrieved_context)}")

            # Step 2: Run Evaluation
            print("-> Running LLM Judges...")
            eval_result = evaluate_single_sample(
                query=question,
                retrieved_context=retrieved_context,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
            )

            # Step 3: Append & Save to file immediately after each question
            results.append(eval_result)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"--> Scores: {eval_result['scores']}")
            print(f"--> Saved progress to '{output_file}'")

        except Exception as e:
            print(f"Error processing question {i + 1}: {e}")

    print(
        f"\n============================================================"
    )
    print(
        f"BENCHMARK COMPLETED! All evaluation results saved to '{output_file}'."
    )


# =====================================================================
# 6. EXECUTION ENTRY POINT
# =====================================================================
from pathlib import Path
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    GROUND_TRUTH_FILE = SCRIPT_DIR.parent / "test_data" / "50_QA_English.json"
    OUTPUT_FILE = SCRIPT_DIR / "rag_evaluation_results.json"

    # Run Async Benchmark
    asyncio.run(
        run_benchmark(
            ground_truth_file=GROUND_TRUTH_FILE,
            output_file=OUTPUT_FILE,
            namespace="b9e49a6e-997f-4273-a698-e59089124af5",  # Adjust namespace as needed
            doc_ids=["d9dd97b8-d90c-40d4-a8b6-cc78642e5e86"],  # Adjust doc IDs as needed
        )
    )