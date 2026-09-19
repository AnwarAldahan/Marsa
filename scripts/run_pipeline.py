"""python scripts/run_pipeline.py 2025-08-15T14:00:00Z"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marsa.pipeline import run

ts = sys.argv[1] if len(sys.argv) > 1 else "2025-08-15T14:00:00Z"
out = run(ts, save="--save" in sys.argv)
print(json.dumps(out["decision_support"], indent=2, ensure_ascii=False))
if "saved_to" in out:
    print("full output:", out["saved_to"])
