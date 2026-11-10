"""Parallel, SHA-256-verified download from the official PyTorch wheel index."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "assets/runtime-wheels"
NAME = "torch-2.7.1+cu126-cp312-cp312-win_amd64.whl"
CHUNK = 16 * 1024 * 1024


class Links(HTMLParser):
    urls = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href")


def main():
    index = "https://download.pytorch.org/whl/cu126/torch/"
    with urllib.request.urlopen(index, timeout=30) as response:
        page = response.read().decode()
    links = Links()
    links.feed(page)
    link = next(url for url in links.urls if urllib.parse.unquote(urllib.parse.urlparse(url).path).endswith("/" + NAME))
    url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(index, link))
    expected_hash = urllib.parse.parse_qs(fragment)["sha256"][0]
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError("Server does not support ranged download")
        total = int(response.headers["Content-Range"].split("/")[-1])
    ROOT.mkdir(parents=True, exist_ok=True)
    partial = ROOT / (NAME + ".partial")
    state_path = ROOT / (NAME + ".download.json")
    completed = set()
    if partial.exists() and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("sha256") == expected_hash:
            completed = set(state["completed"])
    if not partial.exists():
        candidates = sorted(Path(tempfile.gettempdir()).glob("pip-unpack-*/" + NAME), key=lambda p: p.stat().st_mtime, reverse=True)
        resume_bytes = 0
        with partial.open("wb") as output:
            if candidates:
                resume_bytes = candidates[0].stat().st_size // CHUNK * CHUNK
                with candidates[0].open("rb") as source:
                    remaining = resume_bytes
                    while remaining:
                        data = source.read(min(1024 * 1024, remaining))
                        if not data:
                            raise RuntimeError("Incomplete local prefix")
                        output.write(data)
                        remaining -= len(data)
                completed.update(range(resume_bytes // CHUNK))
            output.truncate(total)
        print(f"Reused {resume_bytes / 2**20:.0f} MiB; total {total / 2**20:.0f} MiB", flush=True)
    lock = threading.Lock()
    def fetch(number):
        start = number * CHUNK
        end = min(total - 1, start + CHUNK - 1)
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Unexpected download range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated download range")
                with partial.open("r+b") as output:
                    output.seek(start)
                    output.write(data)
                with lock:
                    completed.add(number)
                    state_path.write_text(json.dumps(dict(sha256=expected_hash, completed=sorted(completed))))
                    print(f"Downloaded {min(total, len(completed) * CHUNK) / total:.0%}", flush=True)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    numbers = [i for i in range((total + CHUNK - 1) // CHUNK) if i not in completed]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, number) for number in numbers]):
            future.result()
    with partial.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected_hash:
        raise RuntimeError("SHA-256 verification failed; wheel was not installed")
    dest = ROOT / NAME
    partial.replace(dest)
    print(f"Verified official SHA-256: {actual}\n{dest}", flush=True)


if __name__ == "__main__":
    main()
