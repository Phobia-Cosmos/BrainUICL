#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

import mne
import numpy as np


URL_TEMPLATE = "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/{sid}.rar"
# TODO:为什么要排除这两个用户？下面分类EEG EOG是什么？这些编号分别代表什么意思？
DEFAULT_SUBJECTS = [sid for sid in range(1, 101) if sid not in (8, 40)]
PICKS_EEG_EOG_A = ["E1-M2", "E2-M1", "F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"]
PICKS_EEG_EOG_B = ["LOC-A2", "ROC-A1", "F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


# TODO:tokens作用是什么？有哪些例子？这里是要做什么？subject指的是被试吗？
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
    # TODO:dict会返回什么？以0-n为key吗？sorted作用又是什么？
    return sorted(dict.fromkeys(subjects))


def download_subject(sid, download_root, force=False):
    # TODO:返回的是一个文件吧？文件的stat都有哪些属性？
    archive = download_root / f"{sid}.rar"
    # TODO:force是什么？
    if archive.exists() and archive.stat().st_size > 0 and not force:
        print(f"Subject {sid}: archive exists, skip download")
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    # TODO:执行下载？这些curl参数都代表什么意思？
    run(["curl", "-L", "--fail", "--no-progress-meter", "--continue-at", "-", URL_TEMPLATE.format(sid=sid), "-o", archive])
    return archive

# TODO:为什么解压出来都是npy文件？
def extract_subject(sid, archive, raw_root, seven_zip, force=False):
    subject_root = raw_root / str(sid)
    rec_path = subject_root / f"{sid}.rec"
    label_path = subject_root / f"{sid}_1.txt"
    if rec_path.exists() and label_path.exists() and not force:
        print(f"Subject {sid}: raw files exist, skip extract")
        return subject_root
    raw_root.mkdir(parents=True, exist_ok=True)
    # TODO:seven_zip会被当作一个工具执行吗？如何确定下载了这个工具？
    run([seven_zip, "x", "-y", archive, f"-o{raw_root}"])
    return subject_root


def read_rec_with_mne(rec_path, tmp_root, verbose):
    tmp_root.mkdir(parents=True, exist_ok=True)
    edf_link = tmp_root / f"{rec_path.stem}.edf"
    if edf_link.exists() or edf_link.is_symlink():
        edf_link.unlink()
    edf_link.symlink_to(rec_path)
    # TODO:mne是什么？为什么要这样读取文件？
    return mne.io.read_raw_edf(edf_link, preload=True, verbose=verbose)


def preprocess_subject(
    sid,
    raw_root,
    output_root,
    tmp_root,
    seq_length=20,
    resample_sfreq=100,
    force=False,
    mimic_author_drop_last=True,
    verbose="WARNING",
):
    subject_root = raw_root / str(sid)
    rec_path = subject_root / f"{sid}.rec"
    label_path = subject_root / f"{sid}_1.txt"
    out_data = output_root / str(sid) / "data"
    out_label = output_root / str(sid) / "label"
    # TODO:out_data / "0.npy"其中使用的是或运算符吗？
    if (out_data / "0.npy").exists() and not force:
        print(f"Subject {sid}: processed npy exists, skip preprocess")
        return
    if not rec_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing raw files for subject {sid}: {rec_path}, {label_path}")

    # TODO:为什么可以直接使用np load label？where判断会得到什么？为什么要判断label是否为4/5 有什么用处 这里不同的label值分别是什么意思？
    labels = np.loadtxt(label_path, dtype=np.int64)
    labels = np.where(labels == 5, 4, labels)

    # TODO:这里读出来的是什么？
    raw = read_rec_with_mne(rec_path, tmp_root, verbose=verbose)

    # TODO:这里会得到什么？各个字段分别是什么意思？
    annotations = mne.Annotations(
        onset=np.arange(labels.shape[0]) * 30.0,
        duration=np.repeat(30.0, labels.shape[0]),
        description=labels.astype(str),
    )
    # TODO:这里把数据处理成什么样子了？为什么还可以直接读取event？原理是什么？
    raw.set_annotations(annotations)

    events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0, verbose=verbose)
    # TODO:raw都有哪些info以及都是什么意思？
    tmax = 30.0 - 1.0 / raw.info["sfreq"]
    # TODO:preload、baseline是什么意思？这个为什么要创建Epochs？
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=verbose,
    )
    # TODO:为什么要重新采样？epochs都有哪些属性？
    epochs.resample(sfreq=resample_sfreq, verbose=verbose)

    # TODO:这里选择的原理是什么？"E1-M2"代表什么？A和B有什么区别吗？
    picks = PICKS_EEG_EOG_A if "E1-M2" in epochs.ch_names else PICKS_EEG_EOG_B
    missing = [ch for ch in picks if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Subject {sid}: missing expected channels {missing}; available={epochs.ch_names}")
    # TODO:意思是将其他的picks删除吗？
    epochs.pick(picks)

    data = epochs.get_data(copy=True).astype(np.float32)
    labels = labels[: data.shape[0]].astype(np.int64)
    seq_num = data.shape[0] // seq_length
    # TODO:save_seq_num是什么作用？mimic_author_drop_last是什么？
    save_seq_num = seq_num - 1 if mimic_author_drop_last else seq_num
    if save_seq_num <= 0:
        raise ValueError(f"Subject {sid}: not enough epochs for seq_length={seq_length}")

    # TODO:shutil是什么？每次都重新创建是吗？
    if force and (output_root / str(sid)).exists():
        shutil.rmtree(output_root / str(sid))
    out_data.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    # TODO:为什么要将一个数据拆成多个文件来存储？
    for idx in range(save_seq_num):
        start = idx * seq_length
        end = start + seq_length
        np.save(out_data / f"{idx}.npy", data[start:end])
        np.save(out_label / f"{idx}.npy", labels[start:end])

    print(
        f"Subject {sid}: saved {save_seq_num} sequences; "
        f"data shape per sequence={(seq_length, data.shape[1], data.shape[2])}"
    )


def main():
    parser = argparse.ArgumentParser(description="Download, extract, and preprocess ISRUC subgroup I for BrainUICL.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subjects, e.g. 1 2 3 or 1-10. Default: 1..100 excluding 8,40.")
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--seven-zip", default="/home/undefined/Disk/ai-storage/BrainUICL/tools/7zip/7zz")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--delete-archive", action="store_true")
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument("--keep-final-sequence", action="store_true", help="Do not mimic the author's seq_num-1 loop.")
    parser.add_argument("--mne-verbose", default="WARNING")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    print("Subjects:", subjects)

    for sid in subjects:
        archive = download_subject(sid, args.download_root, force=args.force_download)
        extract_subject(sid, archive, args.raw_root, args.seven_zip, force=args.force_extract)
        preprocess_subject(
            sid=sid,
            raw_root=args.raw_root,
            output_root=args.output_root,
            tmp_root=args.tmp_root,
            force=args.force_preprocess,
            mimic_author_drop_last=not args.keep_final_sequence,
            verbose=args.mne_verbose,
        )
        if args.delete_archive:
            archive.unlink(missing_ok=True)
        if args.delete_raw:
            shutil.rmtree(args.raw_root / str(sid), ignore_errors=True)


if __name__ == "__main__":
    main()
