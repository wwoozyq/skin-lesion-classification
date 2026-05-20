import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
OUTPUT_ROOT = ROOT / "outputs" / "deep"


def _env(threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_HOME", str(OUTPUT_ROOT / "torch_cache"))
    return env


def _base_args(device):
    return [
        "--model", "mobilenet_v2",
        "--pretrained",
        "--crop",
        "--cache_images",
        "--batch_size", "8",
        "--epochs", "30",
        "--freeze_backbone_epochs", "4",
        "--patience", "8",
        "--device", device,
        "--eval_tta",
    ]


def _experiments(output_root, device):
    base = _base_args(device)
    specs = [
        ("score_mobilenet_160_clean_macro_tta", ["--image_size", "160", "--mask_mode", "clean", "--monitor", "macro_f1"]),
        ("score_mobilenet_192_clean_macro_tta", ["--image_size", "192", "--mask_mode", "clean", "--monitor", "macro_f1"]),
        ("score_mobilenet_224_clean_macro_tta", ["--image_size", "224", "--mask_mode", "clean", "--monitor", "macro_f1"]),
        ("score_mobilenet_192_raw_macro_tta", ["--image_size", "192", "--mask_mode", "raw", "--monitor", "macro_f1"]),
        ("score_mobilenet_224_raw_macro_tta", ["--image_size", "224", "--mask_mode", "raw", "--monitor", "macro_f1"]),
        ("score_mobilenet_192_clean_acc_tta", ["--image_size", "192", "--mask_mode", "clean", "--monitor", "accuracy"]),
        (
            "score_mobilenet_192_clean_light_macro_tta",
            ["--image_size", "192", "--mask_mode", "clean", "--augment_strength", "light", "--monitor", "macro_f1"],
        ),
        (
            "score_mobilenet_192_clean_pad015_macro_tta",
            ["--image_size", "192", "--mask_mode", "clean", "--crop_pad", "0.15", "--monitor", "macro_f1"],
        ),
        (
            "score_mobilenet_192_clean_pad035_macro_tta",
            ["--image_size", "192", "--mask_mode", "clean", "--crop_pad", "0.35", "--monitor", "macro_f1"],
        ),
        (
            "score_mobilenet_192_clean_lr5e5_macro_tta",
            ["--image_size", "192", "--mask_mode", "clean", "--finetune_lr", "5e-5", "--monitor", "macro_f1"],
        ),
    ]
    return [
        {
            "run_name": name,
            "output_dir": output_root / name,
            "args": [*base, *extra, "--output_dir", str(output_root / name)],
        }
        for name, extra in specs
    ]


def _read_metrics(output_dir):
    path = Path(output_dir) / "deep_best_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _write_summary(rows, output_root):
    summary_csv = output_root / "score_search_summary.csv"
    summary_md = output_root / "score_search_summary.md"
    keys = [
        "run_name",
        "status",
        "elapsed_minutes",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "best_score",
        "monitor",
        "image_size",
        "mask_mode",
        "crop_pad",
        "augment_strength",
        "finetune_lr",
        "eval_tta",
        "output_dir",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    completed = [row for row in rows if row.get("status") == "completed"]
    best = max(completed, key=lambda row: float(row.get("accuracy") or 0.0), default=None)
    lines = [
        "# Deep Score Search Summary",
        "",
        "| run | accuracy | macro-F1 | balanced acc | image size | mask | monitor |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {accuracy} | {macro_f1} | {balanced_accuracy} | {image_size} | {mask_mode} | {monitor} |".format(
                run_name=row.get("run_name", ""),
                accuracy=_fmt(row.get("accuracy")),
                macro_f1=_fmt(row.get("macro_f1")),
                balanced_accuracy=_fmt(row.get("balanced_accuracy")),
                image_size=row.get("image_size", ""),
                mask_mode=row.get("mask_mode", ""),
                monitor=row.get("monitor", ""),
            )
        )
    lines.append("")
    if best:
        lines.append(
            f"Best accuracy: `{_fmt(best.get('accuracy'))}` from `{best.get('run_name')}` "
            f"(macro-F1 `{_fmt(best.get('macro_f1'))}`)."
        )
    summary_md.write_text("\n".join(lines) + "\n")


def _fmt(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Data_Proj2")
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT / "score_search"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max_hours", type=float, default=8.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    experiments = _experiments(output_root, args.device)
    if args.dry_run:
        for experiment in experiments:
            cmd = [str(PYTHON), "experiments/train_deep.py", "--data_dir", args.data_dir, *experiment["args"]]
            print(f"\n# {experiment['run_name']}")
            print(" ".join(cmd))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "score_search.log"
    env = _env(args.threads)
    rows = []
    start = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"\nscore search started_at={datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"device={args.device} threads={args.threads} max_hours={args.max_hours:g}\n")
        for experiment in experiments:
            if (time.monotonic() - start) / 3600.0 >= args.max_hours:
                rows.append({"run_name": experiment["run_name"], "status": "skipped_time_limit"})
                continue

            cmd = [str(PYTHON), "experiments/train_deep.py", "--data_dir", args.data_dir, *experiment["args"]]
            run_start = time.monotonic()
            print(f"\n=== {experiment['run_name']} ===", flush=True)
            print(" ".join(cmd), flush=True)
            log.write(f"\n=== {experiment['run_name']} ===\n")
            log.write("command=" + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            returncode = proc.wait()
            metrics = _read_metrics(experiment["output_dir"])
            rows.append({
                "run_name": experiment["run_name"],
                "status": "completed" if returncode == 0 and metrics else "failed",
                "elapsed_minutes": f"{(time.monotonic() - run_start) / 60.0:.2f}",
                "output_dir": str(experiment["output_dir"]),
                **metrics,
            })
            _write_summary(rows, output_root)
        log.write(f"score search finished_at={datetime.now().isoformat(timespec='seconds')}\n")
    _write_summary(rows, output_root)
    print(f"summary={output_root / 'score_search_summary.md'}")


if __name__ == "__main__":
    main()
