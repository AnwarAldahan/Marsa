"""python scripts/run_pipeline.py 2025-08-15T14:00  [--features all]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marsa.pipeline import run
ts = sys.argv[1] if len(sys.argv) > 1 else "2025-08-15T14:00"
fs = sys.argv[sys.argv.index("--features") + 1] if "--features" in sys.argv else "ais_only"
out = run(ts, feature_set=fs)
print(json.dumps(out["recommendation"], indent=2, ensure_ascii=False))
print("full output:", out["saved_to"])
