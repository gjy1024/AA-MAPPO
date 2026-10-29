"""Package every scenario replay losslessly, with reproducible ZIP metadata."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
paths = sorted((ROOT / "artifacts").rglob("seed-*.json"))
index = []
destination = ROOT / "artifacts/replay-archive.zip"
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        metadata = zipfile.ZipInfo(relative, date_time=(2026, 10, 29, 0, 0, 0))
        metadata.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(metadata, content, compresslevel=9)
        index.append({"path":relative, "sha256":hashlib.sha256(content).hexdigest(), "bytes":len(content)})
(ROOT / "artifacts/replay-archive-index.json").write_text(json.dumps({
    "archive":"artifacts/replay-archive.zip", "sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
    "entries":index, "restore":"python -m zipfile -e artifacts/replay-archive.zip .",
    "notes":"All per-scenario trajectories, including failed cases. Summaries, checkpoints and first-scene replays remain directly browsable."},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"replays":len(index), "archive_bytes":destination.stat().st_size}))
