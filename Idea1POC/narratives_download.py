"""Resumable download of first-exposure afni-nosmooth fsaverage6 desc-clean GIFTIs.

Selection uses narratives_prepare.select_scans (first exposure before exclusions).
Each file is fetched to a .part file, size-checked against S3, then renamed.
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from narratives_prepare import cache_path, select_scans

BUCKET = "https://s3.amazonaws.com/fcp-indi"
PREFIX = "data/Projects/narratives/derivatives/afni-nosmooth"
NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def remote_sizes(subject):
    sizes, token = {}, None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={PREFIX}/{subject}/func/"
        if token:
            url += f"&continuation-token={token}"
        root = ElementTree.fromstring(subprocess.run(["curl.exe", "-sSf", url], capture_output=True, check=True).stdout)
        for item in root.findall("s3:Contents", NAMESPACE):
            sizes[item.find("s3:Key", NAMESPACE).text.rsplit("/", 1)[-1]] = int(item.find("s3:Size", NAMESPACE).text)
        token_node = root.find("s3:NextContinuationToken", NAMESPACE)
        if token_node is None:
            return sizes
        token = token_node.text


def fetch(destination, size):
    if destination.is_file() and destination.stat().st_size == size:
        return "present"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"{BUCKET}/{PREFIX}/{destination.parent.name}/func/{destination.name}"
    subprocess.run(["curl.exe", "-sSfL", "--retry", "5", "-C", "-", url, "-o", str(partial)], check=True)
    if partial.stat().st_size != size:
        raise ValueError(f"Size mismatch: {destination.name}")
    partial.replace(destination)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "narratives_meta")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    metadata = args.metadata_dir
    selected, _ = select_scans(json.loads((metadata / "task_meta.json").read_text()),
                               json.loads((metadata / "scan_exclude.json").read_text()), args.tasks)
    cache = metadata / "processed_cache"
    sizes = {}
    jobs = []
    for scan in selected:
        if scan["subject"] not in sizes:
            sizes[scan["subject"]] = remote_sizes(scan["subject"])
        for hemisphere in ("L", "R"):
            destination = cache_path(cache, scan, hemisphere)
            if destination.name not in sizes[scan["subject"]]:
                raise ValueError(f"Not on S3: {destination.name}")
            jobs.append((destination, sizes[scan["subject"]][destination.name]))
    total = sum(size for _, size in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.1f} GB", flush=True)
    done, failures = 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, destination, size): destination for destination, size in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as error:  # report every failure; rerun resumes
                failures.append(f"{futures[future].name}: {error}")
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} complete, {len(failures)} failed", flush=True)
    for failure in failures:
        print("FAILED", failure, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
