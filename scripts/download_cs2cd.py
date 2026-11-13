"""Download only the JSON event files of the CS2CD dataset (~1.8 GB, skips ~51 GB of tick parquet).

Source: https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection (CC BY 4.0)
Usage:  python scripts/download_cs2cd.py [--out data/raw/cs2cd] [--workers 8]
Re-running is safe: files already present with the correct size are skipped.
"""
import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
FOLDERS = ["with_cheater_present", "no_cheater_present"]
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{{folder}}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def list_json_files():
    files = []
    for folder in FOLDERS:
        with urllib.request.urlopen(API.format(folder=folder), timeout=60) as r:
            entries = json.load(r)
        files += [(e["path"], e["size"]) for e in entries if e["path"].endswith(".json")]
    return files


def download(path, size, out_dir):
    dest = out_dir / path
    if dest.exists() and dest.stat().st_size == size:
        return path, "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(RAW.format(path=path), timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            return path, "ok"
        except Exception as e:  # retry on network errors
            err = e
    tmp.unlink(missing_ok=True)
    return path, f"FAILED: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/cs2cd")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out)

    files = list_json_files()
    total = sum(s for _, s in files)
    print(f"{len(files)} JSON files, {total / 1e9:.2f} GB", flush=True)

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download, p, s, out_dir) for p, s in files]
        for i, fut in enumerate(as_completed(futures), 1):
            path, status = fut.result()
            if status.startswith("FAILED"):
                failed.append((path, status))
            if i % 50 == 0 or i == len(files):
                print(f"{i}/{len(files)} done", flush=True)

    if failed:
        print(f"{len(failed)} failed (re-run to retry):")
        for p, s in failed:
            print(" ", p, s)
        sys.exit(1)
    print("All files downloaded and size-verified.")


if __name__ == "__main__":
    main()
