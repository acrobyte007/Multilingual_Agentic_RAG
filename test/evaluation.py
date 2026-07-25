import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================


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


# ==========================================
# 2. SYSTEM PROMPTS & LLM SETUP
# ==========================================

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

# Base Gemini Model Setup
base_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.0,  # Zero temperature for deterministic evaluation
    max_tokens=None,
    timeout=None,
    max_retries=1,
)

# Metric-specific models with structured output bindings
precision_model = base_model.with_structured_output(ContextPrecisionResponse)
recall_model = base_model.with_structured_output(ContextRecallResponse)
faithfulness_model = base_model.with_structured_output(FaithfulnessResponse)


# ==========================================
# 3. MATHEMATICAL CALCULATION FUNCTIONS
# ==========================================


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


# ==========================================
# 4. SINGLE EVALUATION WORKFLOW
# ==========================================


def evaluate_single_sample(
    query: str,
    retrieved_context: List[str],
    generated_answer: str,
    ground_truth: str,
) -> Dict:
    """Runs all 3 evaluations on a single query instance."""
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
    # Ensure ordered binary sequence
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

    # Return compiled result dictionary
    return {
        "query": query,
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


# ==========================================
# 5. BATCH EVALUATOR WITH IMMEDIATE SAVING
# ==========================================


def run_batch_evaluation(
    eval_dataset: List[Dict], output_filepath: str = "evaluation_results.json"
):
    """Evaluates samples sequentially and saves to file immediately after each sample."""
    # Load existing progress if file exists
    results = []
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(
                f"Resuming evaluation. Found {len(results)} previously completed samples."
            )
        except json.JSONDecodeError:
            results = []

    start_index = len(results)

    for i in range(start_index, len(eval_dataset)):
        sample = eval_dataset[i]
        print(
            f"Evaluating sample {i + 1}/{len(eval_dataset)}: '{sample['query'][:40]}...'"
        )

        # Run evaluation
        eval_result = evaluate_single_sample(
            query=sample["query"],
            retrieved_context=sample["retrieved_context"],
            generated_answer=sample["generated_answer"],
            ground_truth=sample["ground_truth"],
        )

        # Append result
        results.append(eval_result)

        # Save immediately to output JSON file
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(
            f"Saved sample {i + 1} to {output_filepath} -> Scores: {eval_result['scores']}"
        )

    print(
        f"\nEvaluation Complete! All results written to '{output_filepath}'."
    )


# ==========================================
# 6. EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    test_dataset = [
        {
            "query": "Where was Albert Einstein born and when did he win the Nobel Prize?",
            "retrieved_context": [
                "Albert Einstein was born in Ulm, Kingdom of Württemberg, German Empire, on 14 March 1879.",
                "He moved to Switzerland in 1895 and gained his diploma at the Federal Polytechnic School in Zurich.",
                "Einstein received the 1921 Nobel Prize in Physics for his services to Theoretical Physics.",
            ],
            "generated_answer": "Einstein was born in Ulm, Germany in 1879. He won the Nobel Prize in Physics in 1921.",
            "ground_truth": "Albert Einstein was born in Ulm, Germany in 1879 and won the Nobel Prize in Physics in 1921.",
        },
        {
            "query": "What is the capital of France and its population?",
            "retrieved_context": [
                "Paris is the capital and most populous city of France.",
                "The city covers an area of 105 square kilometers.",
            ],
            "generated_answer": "Paris is the capital of France with a population of 10 million.",
            "ground_truth": "The capital of France is Paris and its population is over 2 million.",
        },
    ]

    # Run batch process with real-time saving
    run_batch_evaluation(
        test_dataset, output_filepath="rag_eval_results.json"
    )