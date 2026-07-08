#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


URL_TEMPLATE = "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/{sid}.rar"
DEFAULT_SUBJECTS = [sid for sid in range(1, 101) if sid not in (8, 40)]


def parse_subjects(tokens):
    if not tokens:
        return DEFAULT_SUBJECTS
    subjects = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                subjects.extend(range(int(start), int(end) + 1))
            else:
                subjects.append(int(part))
    return [sid for sid in sorted(dict.fromkeys(subjects)) if sid not in (8, 40)]


def remote_size(url):
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def split_ranges(size, segments):
    chunk = (size + segments - 1) // segments
    ranges = []
    start = 0
    while start < size:
        end = min(size - 1, start + chunk - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def run_curl_range(url, start, end, output):
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-time",
        "90",
        "--speed-limit",
        "1024",
        "-r",
        f"{start}-{end}",
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    expected = end - start + 1
    actual = tmp.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{output.name}: expected {expected} bytes, got {actual}")
    tmp.rename(output)


def download_subject(sid, download_root, part_root, segments):
    url = URL_TEMPLATE.format(sid=sid)
    target = download_root / f"{sid}.rar"
    size = remote_size(url)

    if target.exists() and target.stat().st_size == size:
        print(f"Subject {sid}: archive complete, skip ({size} bytes)", flush=True)
        return sid, size

    if target.exists():
        print(f"Subject {sid}: removing incomplete archive {target.stat().st_size}/{size}", flush=True)
        target.unlink()

    subject_part_root = part_root / str(sid)
    if subject_part_root.exists():
        shutil.rmtree(subject_part_root)
    subject_part_root.mkdir(parents=True, exist_ok=True)
    ranges = split_ranges(size, segments)
    print(f"Subject {sid}: downloading {size} bytes in {len(ranges)} segments", flush=True)

    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = []
        for idx, (start, end) in enumerate(ranges):
            output = subject_part_root / f"{idx:03d}.part"
            futures.append(pool.submit(run_curl_range, url, start, end, output))
        for future in as_completed(futures):
            future.result()

    tmp_target = target.with_suffix(".rar.tmp")
    tmp_target.unlink(missing_ok=True)
    with tmp_target.open("wb") as writer:
        for idx in range(len(ranges)):
            part = subject_part_root / f"{idx:03d}.part"
            with part.open("rb") as reader:
                shutil.copyfileobj(reader, writer)
    actual = tmp_target.stat().st_size
    if actual != size:
        raise RuntimeError(f"Subject {sid}: assembled size mismatch {actual}/{size}")
    tmp_target.rename(target)
    shutil.rmtree(subject_part_root)
    print(f"Subject {sid}: archive complete ({size} bytes)", flush=True)
    return sid, size


def main():
    parser = argparse.ArgumentParser(description="Parallel ranged downloader for ISRUC subgroup-I .rar archives.")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--part-root", type=Path, default=Path("/home/undefined/Disk/ai-storage/BrainUICL/tmp/isruc_parts"))
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--file-workers", type=int, default=2)
    args = parser.parse_args()

    args.download_root.mkdir(parents=True, exist_ok=True)
    args.part_root.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    print("Subjects:", subjects, flush=True)
    print(f"file_workers={args.file_workers} segments={args.segments}", flush=True)

    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futures = [pool.submit(download_subject, sid, args.download_root, args.part_root, args.segments)
                   for sid in subjects]
        for future in as_completed(futures):
            sid, size = future.result()
            print(f"Done subject {sid}: {size} bytes", flush=True)


if __name__ == "__main__":
    main()
