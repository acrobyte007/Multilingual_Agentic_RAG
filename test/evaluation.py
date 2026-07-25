import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from database.vector_database import pinecone_service
from services.agent import get_rag_answer

load_dotenv()

# Initialize Pinecone
pinecone_service.initialize()

# =====================================================================
# 1. PYDANTIC SCHEMAS
# =====================================================================


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

base_eval_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

precision_model = base_eval_model.with_structured_output(
    ContextPrecisionResponse
)
recall_model = base_eval_model.with_structured_output(ContextRecallResponse)
faithfulness_model = base_eval_model.with_structured_output(
    FaithfulnessResponse
)


# =====================================================================
# 3. RATE-LIMIT HANDLING & RETRY UTILITY
# =====================================================================


async def invoke_with_rate_limit_retry(
    chain, inputs: dict, max_retries: int = 5, initial_delay: float = 10.0
):
    """Executes a LangChain runnable and automatically waits/retries if a 429 Rate Limit error is hit."""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Try to extract the wait time suggested by Google API (e.g., 'retry in 23.5s')
                match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 2.0
                else:
                    wait_time = delay

                print(
                    f"\n⚠️ Rate Limit (429) hit on attempt {attempt}/{max_retries}. Pausing for {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
                delay *= 1.5  # Exponential backoff
            else:
                # Raise non-rate-limit exceptions directly
                raise e

    raise Exception(f"Failed after {max_retries} rate-limit retries.")


# =====================================================================
# 4. MATHEMATICAL CALCULATION FUNCTIONS
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
# 5. SINGLE SAMPLE EVALUATION ENGINE (ASYNC WITH RETRIES)
# =====================================================================


async def evaluate_single_sample(
    query: str,
    retrieved_context: List[str],
    generated_answer: str,
    ground_truth: str,
) -> Dict:
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
    prec_res: ContextPrecisionResponse = await invoke_with_rate_limit_retry(
        prec_prompt | precision_model,
        {
            "query": query,
            "ground_truth": ground_truth,
            "context": formatted_context,
        },
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
    rec_res: ContextRecallResponse = await invoke_with_rate_limit_retry(
        rec_prompt | recall_model,
        {"ground_truth": ground_truth, "context": formatted_context},
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
    faith_res: FaithfulnessResponse = await invoke_with_rate_limit_retry(
        faith_prompt | faithfulness_model,
        {"answer": generated_answer, "context": formatted_context},
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
# 6. FILE SAVE HELPER
# =====================================================================


def save_results_to_file(results_list: list, output_filepath: str):
    """Saves output to disk and forces immediate flush."""
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


# =====================================================================
# 7. BENCHMARK PIPELINE
# =====================================================================


async def run_benchmark(
    ground_truth_file: str,
    output_file: str = "evaluation_output.json",
    namespace: str = "anemia_research",
    doc_ids: List[str] = ["doc_001"],
    delay_between_questions: float = 3.0,  # Pacing pause
):
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(
            f"Ground truth dataset '{ground_truth_file}' not found."
        )

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(
                f"Resuming evaluation: {len(results)} samples already completed in '{output_file}'."
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
            # Step 1: UNPACK TUPLE (answer: str, chunks: List[str])
            rag_output = await get_rag_answer(
                namespace=namespace,
                query=question,
                doc_ids=doc_ids,
            )

            if isinstance(rag_output, tuple):
                generated_answer, retrieved_context = rag_output
            else:
                generated_answer = rag_output
                retrieved_context = []

            # Format retrieved_context safely
            if isinstance(retrieved_context, str):
                retrieved_context = [retrieved_context]
            elif not retrieved_context:
                retrieved_context = [
                    "No context chunks were retrieved by the agent."
                ]

            print(f"-> Answer Generated: {str(generated_answer)[:80]}...")
            print(f"-> Chunks Retrieved: {len(retrieved_context)}")

            # Step 2: Run Evaluation with Rate Limit Retries
            print("-> Running LLM Judges...")
            eval_result = await evaluate_single_sample(
                query=question,
                retrieved_context=retrieved_context,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
            )

            # Step 3: Save results sample-by-sample
            results.append(eval_result)
            save_results_to_file(results, output_file)

            print(f"--> Scores: {eval_result['scores']}")
            print(
                f"--> Saved sample [{i + 1}/{len(ground_truth_data)}] to '{output_file}'"
            )

        except Exception as e:
            print(
                f"❌ Error on Question [{i + 1}/{len(ground_truth_data)}]: {e}"
            )

            error_entry = {
                "query": question,
                "generated_answer": f"ERROR: {str(e)}",
                "ground_truth": ground_truth,
                "scores": {
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "faithfulness": 0.0,
                },
                "details": {"error": str(e)},
            }

            results.append(error_entry)
            save_results_to_file(results, output_file)

        # Pause slightly between questions to avoid hitting RPM limits
        if delay_between_questions > 0:
            await asyncio.sleep(delay_between_questions)

    print(
        f"\n============================================================"
    )
    print(
        f"BENCHMARK COMPLETED! All evaluation results saved to '{output_file}'."
    )


# =====================================================================
# 8. MAIN ENTRY POINT
# =====================================================================


async def main():
    SCRIPT_DIR = Path(__file__).resolve().parent

    GROUND_TRUTH_FILE = SCRIPT_DIR.parent / "test_data" / "50_QA_English.json"
    OUTPUT_FILE = SCRIPT_DIR / "rag_evaluation_results.json"

    # Initialize all services in the exact same event loop
    pinecone_service.initialize()

    # If embedding_service has an async initialize method:
    # await embedding_service.initialize()

    await run_benchmark(
        ground_truth_file=str(GROUND_TRUTH_FILE),
        output_file=str(OUTPUT_FILE),
        namespace="b9e49a6e-997f-4273-a698-e59089124af5",
        doc_ids=["d9dd97b8-d90c-40d4-a8b6-cc78642e5e86"],
        delay_between_questions=3.0,  # 3 seconds pause between each item
    )


if __name__ == "__main__":
    asyncio.run(main())
