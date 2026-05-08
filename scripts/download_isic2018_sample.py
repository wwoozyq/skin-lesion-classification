import argparse
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
}

IMAGE_URL_TEMPLATE = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"


def select_samples(ground_truth_csv, max_per_class, seed):
    df = pd.read_csv(ground_truth_csv)
    required = {"image", *LABEL_MAP.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ground truth CSV: {sorted(missing)}")

    selected = []
    for external_label, project_label in LABEL_MAP.items():
        class_df = df[df[external_label] == 1].copy()
        class_df = class_df.sample(
            n=min(max_per_class, len(class_df)),
            random_state=seed,
        )
        for _, row in class_df.iterrows():
            selected.append({
                "image_id": row["image"],
                "dx": project_label,
                "external_label": external_label,
                "source": "ISIC2018_Task3_Training",
                "image_file": f"{row['image']}.jpg",
                "has_mask": False,
            })

    manifest = pd.DataFrame(selected).sort_values(["dx", "image_id"]).reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No MEL/NV/VASC rows were selected.")
    return manifest


def download_with_python(url, output_path):
    with urllib.request.urlopen(url, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if "image" not in content_type and len(data) < 1024:
        raise ValueError(f"Unexpected response: {content_type}")
    output_path.write_bytes(data)


def download_with_curl(url, output_path):
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    result = subprocess.run(
        ["curl", "-L", "--fail", "--noproxy", "*", url, "-o", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    if not output_path.exists() or output_path.stat().st_size <= 1024:
        raise ValueError("Downloaded file is unexpectedly small.")


def download_one(image_id, output_path, retries, sleep_seconds, method):
    if output_path.exists() and output_path.stat().st_size > 1024:
        return "exists"

    url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if method == "curl":
                download_with_curl(url, output_path)
            else:
                download_with_python(url, output_path)
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {image_id}: {last_error}")


def write_metadata(manifest, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    manifest[["image_id", "dx"]].to_csv(output_dir / "label.csv", index=False)
    summary = manifest["dx"].value_counts().rename_axis("dx").reset_index(name="count")
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument("--method", default="curl", choices=["curl", "python"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = select_samples(args.ground_truth_csv, args.max_per_class, args.seed)
    summary = write_metadata(manifest, output_dir)

    counts = {"downloaded": 0, "exists": 0}
    for row in manifest.itertuples(index=False):
        output_path = image_dir / row.image_file
        status = download_one(
            row.image_id,
            output_path,
            args.retries,
            args.sleep_seconds,
            args.method,
        )
        counts[status] += 1
        print(f"{status}: {row.image_id} -> {output_path}")

    print("Summary:")
    print(summary.to_string(index=False))
    print(f"Downloaded: {counts['downloaded']}")
    print(f"Already existed: {counts['exists']}")
    print(f"Saved label CSV: {output_dir / 'label.csv'}")
    print(f"Saved manifest: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
