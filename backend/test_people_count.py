import json
import traceback
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from backend.app import analyze_video

    video_path = Path(__file__).resolve().parent / "uploads" / "sample video 2.mp4"
    result = analyze_video(str(video_path))

    out_path = Path(__file__).resolve().parent / "cv_check.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("peopleCount:", result.get("peopleCount"))
    print("averagePeopleCount:", result.get("averagePeopleCount"))
    print("maximumPeopleCount:", result.get("maximumPeopleCount"))
    print("totalFramesProcessed:", result.get("totalFramesProcessed"))
    print("debugFramePath:", result.get("debugFramePath"))

    for key, value in sorted(result.items()):
        if key in {"peopleCount", "averagePeopleCount", "maximumPeopleCount", "totalFramesProcessed", "debugFramePath"}:
            continue
        print(f"{key}: {value}")

    print("JSON saved to:", out_path)
except Exception:
    traceback.print_exc()
    raise
