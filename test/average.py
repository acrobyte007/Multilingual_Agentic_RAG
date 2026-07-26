from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent

RESULT_FILE = SCRIPT_DIR / "rag_evaluation_results_2.json"

with open(RESULT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

num_samples = len(results)

avg_precision = sum(
    r["scores"]["context_precision"] for r in results
) / num_samples

avg_recall = sum(
    r["scores"]["context_recall"] for r in results
) / num_samples

avg_faithfulness = sum(
    r["scores"]["faithfulness"] for r in results
) / num_samples

print(f"Total Samples      : {num_samples}")
print(f"Context Precision  : {avg_precision:.4f}")
print(f"Context Recall     : {avg_recall:.4f}")
print(f"Faithfulness       : {avg_faithfulness:.4f}")