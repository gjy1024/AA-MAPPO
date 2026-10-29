"""Validate tracked site links, data and artifact checksums without a browser."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for name in ("src", "href"):
            if attrs.get(name):
                self.references.append(attrs[name])


def main():
    root = Path(__file__).resolve().parents[1]
    site = root / "site"
    parser = References()
    parser.feed((site / "index.html").read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML IDs"
    for ref in parser.references:
        parsed = urlsplit(ref)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (site / unquote(parsed.path)).resolve()
        assert site.resolve() in target.parents, f"Out-of-site reference: {ref}"
        assert target.is_file(), f"Missing site reference: {ref}"
    for path in (site / "data").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        assert data.get("schema_version") == "1.0", f"Schema version missing: {path.name}"
        if data.get("kind") == "evaluation":
            assert len(data["trajectory"]) > 1
            times = [frame["time"] for frame in data["trajectory"]]
            assert all(b > a for a, b in zip(times, times[1:])), "Unordered replay time"
            assert data["provenance"].get("paper_reproduction") is not True, "Do not mislabel maintenance/demo as paper reproduction"
    manifest = json.loads((site / "data/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["runs"] + manifest["artifacts"]:
        if item.get("path"):
            assert (site / item["path"]).is_file(), f"Missing manifest entry: {item['path']}"
    figures = json.loads((site / "data/historical-figures.json").read_text(encoding="utf-8"))
    for item in figures["figures"]:
        path = site / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "Figure checksum mismatch"
    inventory = json.loads((root / "artifacts/legacy/inventory.json").read_text(encoding="utf-8"))
    for item in inventory["artifacts"]:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"], "Historical artifact checksum mismatch"
    print(f"PASS: site references, unique IDs, JSON, manifest and {len(inventory['artifacts'])} historical artifact checksums")


if __name__ == "__main__":
    main()
