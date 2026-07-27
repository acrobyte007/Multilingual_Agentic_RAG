from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FILE = SCRIPT_DIR / "rag_evaluation_results_2.json"

with open(RESULT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Remove samples where all metrics are zero
filtered_results = [
    r for r in results
    if not (
        r["scores"]["context_precision"] == 0
        and r["scores"]["context_recall"] == 0
        and r["scores"]["faithfulness"] == 0
    )
]

discarded = len(results) - len(filtered_results)

if len(filtered_results) == 0:
    print("No valid samples left after filtering.")
else:
    num_samples = len(filtered_results)

    avg_precision = sum(
        r["scores"]["context_precision"] for r in filtered_results
    ) / num_samples

    avg_recall = sum(
        r["scores"]["context_recall"] for r in filtered_results
    ) / num_samples

    avg_faithfulness = sum(
        r["scores"]["faithfulness"] for r in filtered_results
    ) / num_samples

    print(f"Original Samples   : {len(results)}")
    print(f"Discarded Samples  : {discarded}")
    print(f"Valid Samples      : {num_samples}")
    print(f"Context Precision  : {avg_precision:.4f}")
    print(f"Context Recall     : {avg_recall:.4f}")
    print(f"Faithfulness       : {avg_faithfulness:.4f}")