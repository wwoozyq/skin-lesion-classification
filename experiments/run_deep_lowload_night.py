import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "deep"
SUMMARY_COLUMNS = [
    "run_name",
    "status",
    "returncode",
    "started_at",
    "finished_at",
    "elapsed_minutes",
    "model",
    "monitor",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "best_score",
    "output_dir",
]


def _thread_limited_env(num_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    env["NUMEXPR_NUM_THREADS"] = str(num_threads)
    env.setdefault("TORCH_HOME", str(DEFAULT_OUTPUT_ROOT / "torch_cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _maybe_reexec_with_caffeinate(args):
    if not args.caffeinate:
        return
    if os.environ.get("LOWLOAD_NIGHT_CAFFEINATED") == "1":
        return
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        print("warning: caffeinate not found; continuing without sleep prevention", flush=True)
        return

    env = os.environ.copy()
    env["LOWLOAD_NIGHT_CAFFEINATED"] = "1"
    cmd = [caffeinate, "-dimsu", sys.executable, *sys.argv]
    print("restarting under caffeinate:", " ".join(cmd), flush=True)
    os.execve(caffeinate, cmd, env)


def _experiments(output_root):
    common = [
        "--model", "mobilenet_v2",
        "--pretrained",
        "--crop",
        "--cache_images",
        "--image_size", "160",
        "--batch_size", "8",
        "--epochs", "25",
        "--freeze_backbone_epochs", "4",
        "--patience", "6",
        "--mask_mode", "clean",
    ]
    return [
        {
            "run_name": "night_lowload_mobilenet_clean_acc",
            "reason": "primary clean-mask accuracy run",
            "args": [
                *common,
                "--monitor", "accuracy",
                "--output_dir", str(output_root / "night_lowload_mobilenet_clean_acc"),
            ],
        },
        {
            "run_name": "night_lowload_mobilenet_clean_light_acc",
            "reason": "light-augmentation control",
            "args": [
                *common,
                "--augment_strength", "light",
                "--monitor", "accuracy",
                "--output_dir", str(output_root / "night_lowload_mobilenet_clean_light_acc"),
            ],
        },
        {
            "run_name": "night_lowload_mobilenet_clean_macro_f1",
            "reason": "macro-F1 monitor control",
            "args": [
                *common,
                "--monitor", "macro_f1",
                "--output_dir", str(output_root / "night_lowload_mobilenet_clean_macro_f1"),
            ],
        },
    ]


def _read_metrics(output_dir):
    metrics_path = Path(output_dir) / "deep_best_metrics.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _write_summary(rows, summary_csv, summary_md, target_accuracy):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})

    completed = [row for row in rows if row.get("status") == "completed"]
    best_acc = None
    if completed:
        best_acc = max(completed, key=lambda row: float(row.get("accuracy") or 0.0))

    lines = [
        "# Low-Load CPU Deep Learning Night Run",
        "",
        f"Target validation accuracy: `{target_accuracy:.2f}`",
        "",
        "| run | status | accuracy | macro-F1 | balanced accuracy | elapsed min |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {status} | {accuracy} | {macro_f1} | {balanced_accuracy} | {elapsed_minutes} |".format(
                run_name=row.get("run_name", ""),
                status=row.get("status", ""),
                accuracy=_fmt(row.get("accuracy")),
                macro_f1=_fmt(row.get("macro_f1")),
                balanced_accuracy=_fmt(row.get("balanced_accuracy")),
                elapsed_minutes=_fmt(row.get("elapsed_minutes")),
            )
        )
    lines.append("")
    if best_acc:
        reached = float(best_acc.get("accuracy") or 0.0) >= target_accuracy
        lines.append(
            "Best accuracy run: `{}` with accuracy `{}`. Target reached: `{}`.".format(
                best_acc.get("run_name", ""),
                _fmt(best_acc.get("accuracy")),
                reached,
            )
        )
    else:
        lines.append("No completed run produced `deep_best_metrics.csv`.")
    lines.append("")
    lines.append("Use this as a report/presentation extension only; the main quantitative submission remains traditional ML.")
    summary_md.write_text("\n".join(lines) + "\n")


def _fmt(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _run_one(experiment, args, env, log_handle):
    cmd = [str(PYTHON), "experiments/train_deep.py", "--data_dir", args.data_dir, *experiment["args"]]
    started = datetime.now()
    start_time = time.monotonic()
    print(f"\n=== {experiment['run_name']} ===", flush=True)
    print(f"reason: {experiment['reason']}", flush=True)
    print("command:", " ".join(cmd), flush=True)
    log_handle.write(f"\n=== {experiment['run_name']} ===\n")
    log_handle.write(f"started_at={started.isoformat(timespec='seconds')}\n")
    log_handle.write("command=" + " ".join(cmd) + "\n")
    log_handle.flush()

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
        log_handle.write(line)
        log_handle.flush()
    returncode = proc.wait()
    finished = datetime.now()
    elapsed_minutes = (time.monotonic() - start_time) / 60.0
    output_dir = experiment["args"][experiment["args"].index("--output_dir") + 1]
    metrics = _read_metrics(output_dir)
    status = "completed" if returncode == 0 and metrics else "failed"
    row = {
        "run_name": experiment["run_name"],
        "status": status,
        "returncode": returncode,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "elapsed_minutes": f"{elapsed_minutes:.2f}",
        "output_dir": output_dir,
        **metrics,
    }
    log_handle.write(f"finished_at={row['finished_at']} returncode={returncode} status={status}\n")
    log_handle.flush()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Data_Proj2")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--target_accuracy", type=float, default=0.65)
    parser.add_argument("--max_hours", type=float, default=8.0)
    parser.add_argument("--caffeinate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    _maybe_reexec_with_caffeinate(args)

    output_root = Path(args.output_root)
    experiments = _experiments(output_root)
    if args.dry_run:
        print(f"threads={args.threads} target_accuracy={args.target_accuracy:.2f} max_hours={args.max_hours:g}")
        for experiment in experiments:
            cmd = [str(PYTHON), "experiments/train_deep.py", "--data_dir", args.data_dir, *experiment["args"]]
            print(f"\n# {experiment['run_name']}")
            print(" ".join(cmd))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "night_lowload_cpu_run.log"
    summary_csv = output_root / "night_lowload_summary.csv"
    summary_md = output_root / "night_lowload_summary.md"
    env = _thread_limited_env(args.threads)

    print(f"low-load deep night run started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"threads={args.threads} target_accuracy={args.target_accuracy:.2f} max_hours={args.max_hours:g}")
    print(f"log={log_path}")

    start_time = time.monotonic()
    rows = []
    with log_path.open("a") as log_handle:
        log_handle.write("\n\n")
        log_handle.write(f"low-load deep night run started_at={datetime.now().isoformat(timespec='seconds')}\n")
        log_handle.write(f"threads={args.threads} target_accuracy={args.target_accuracy:.2f} max_hours={args.max_hours:g}\n")
        stop_after_light_control = False
        index = 0
        while index < len(experiments):
            experiment = experiments[index]
            if index == 2 and (time.monotonic() - start_time) / 3600.0 >= args.max_hours:
                rows.append({
                    "run_name": experiment["run_name"],
                    "status": "skipped_time_limit",
                    "output_dir": experiment["args"][experiment["args"].index("--output_dir") + 1],
                })
                index += 1
                continue
            if index == 2 and stop_after_light_control:
                rows.append({
                    "run_name": experiment["run_name"],
                    "status": "skipped_primary_reached_target",
                    "output_dir": experiment["args"][experiment["args"].index("--output_dir") + 1],
                })
                index += 1
                continue

            row = _run_one(experiment, args, env, log_handle)
            rows.append(row)
            _write_summary(rows, summary_csv, summary_md, args.target_accuracy)

            if index == 0 and row.get("status") == "completed":
                first_accuracy = float(row.get("accuracy") or 0.0)
                if first_accuracy >= args.target_accuracy:
                    log_handle.write(
                        f"primary run reached target accuracy {first_accuracy:.4f}; "
                        "will run only the light-augmentation control and stop before macro-F1 control.\n"
                    )
                    stop_after_light_control = True
            index += 1

        _write_summary(rows, summary_csv, summary_md, args.target_accuracy)
        log_handle.write(f"low-load deep night run finished_at={datetime.now().isoformat(timespec='seconds')}\n")

    print(f"summary_csv={summary_csv}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
