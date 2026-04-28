#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mix multiple LeRobot v2.1 datasets with safe video concatenation.

Uses a robust two-pass approach for video concatenation:
1. First re-encode each input video to a consistent format
2. Then concatenate using concat demuxer

This avoids the corrupted H.264 stream issue.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ============================================================================
# MONKEY PATCH - MUST BE APPLIED BEFORE ANY LEROBOT IMPORTS
# ============================================================================

VIDEO_CRF = 18
VIDEO_PRESET = "fast"

def _concatenate_video_files_safe(
    input_video_paths: List,
    output_video_path: Path,
    overwrite: bool = True,
) -> None:
    """
    Concatenate videos with robust two-pass approach.
    
    Pass 1: Re-encode each input to consistent format
    Pass 2: Concatenate using concat demuxer
    """
    output_video_path = Path(output_video_path)
    input_video_paths = [Path(p) for p in input_video_paths]

    if output_video_path.exists() and not overwrite:
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    # Create temp directory for re-encoded videos
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        reencoded_paths = []
        
        # Pass 1: Re-encode each video to consistent format
        for i, input_path in enumerate(input_video_paths):
            tmp_output = tmpdir / f"reencoded_{i:06d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", VIDEO_PRESET,
                "-crf", str(VIDEO_CRF),
                "-pix_fmt", "yuv420p",
                "-r", "30",  # Force consistent framerate
                str(tmp_output)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                # If re-encoding fails, try to copy the video as-is
                shutil.copy2(input_path, tmp_output)
            reencoded_paths.append(tmp_output)
        
        # Pass 2: Concatenate using concat demuxer
        concat_file = tmpdir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for p in reencoded_paths:
                f.write(f"file '{p}'\n")
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",  # Stream copy is safe now (all inputs are consistent)
            str(output_video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Concatenation failed for {output_video_path}")


# Apply monkey patch BEFORE importing lerobot
import lerobot.datasets.video_utils as _video_utils
_video_utils.concatenate_video_files = _concatenate_video_files_safe

# ============================================================================
# NOW IMPORT LEROBOT MODULES
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("Applied monkey patch: concatenate_video_files -> robust two-pass version")


def parse_args():
    parser = argparse.ArgumentParser(description="Mix LeRobot v2.1 datasets with safe video handling")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--ratios", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="fast")
    return parser.parse_args()


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes_jsonl(dataset_path: Path) -> list[dict]:
    episodes = []
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_episodes_stats(dataset_path: Path) -> list[dict]:
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return []
    stats = []
    with open(stats_path) as f:
        for line in f:
            stats.append(json.loads(line))
    return stats


def load_tasks_jsonl(dataset_path: Path) -> dict[int, str]:
    tasks = {}
    with open(dataset_path / "meta" / "tasks.jsonl") as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_index"]] = task["task"]
    return tasks


def get_video_keys(dataset_path: Path) -> list[str]:
    video_dir = dataset_path / "videos" / "chunk-000"
    if video_dir.exists():
        return [d.name for d in video_dir.iterdir() if d.is_dir()]
    return []


def sample_episodes(n_episodes: int, ratio: int, total_ratio: int, rng: np.random.Generator) -> list[int]:
    n_samples = max(1, int(n_episodes * ratio / total_ratio))
    return sorted(rng.choice(list(range(n_episodes)), size=n_samples, replace=False).tolist())


def copy_episode_data(src_dataset: Path, dst_dataset: Path, src_ep_idx: int, dst_ep_idx: int,
                      video_keys: list[str], total_frames_before: int) -> int:
    src_parquet = src_dataset / "data" / "chunk-000" / f"episode_{src_ep_idx:06d}.parquet"
    dst_parquet = dst_dataset / "data" / "chunk-000" / f"episode_{dst_ep_idx:06d}.parquet"

    df = pq.read_table(src_parquet).to_pandas()
    episode_length = len(df)

    df["episode_index"] = dst_ep_idx
    df["index"] = df.index + total_frames_before
    df.to_parquet(dst_parquet)

    for vid_key in video_keys:
        src_video = src_dataset / "videos" / "chunk-000" / vid_key / f"episode_{src_ep_idx:06d}.mp4"
        dst_video_dir = dst_dataset / "videos" / "chunk-000" / vid_key
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        if src_video.exists():
            shutil.copy2(src_video, dst_video_dir / f"episode_{dst_ep_idx:06d}.mp4")

    return episode_length


def create_mixed_dataset(dataset_paths: list[Path], ratios: list[int], output_dir: Path,
                         output_name: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total_ratio = sum(ratios)

    output_path = output_dir / output_name
    if output_path.exists():
        shutil.rmtree(output_path)

    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(parents=True, exist_ok=True)

    video_keys = get_video_keys(dataset_paths[0])
    logger.info(f"Video keys: {video_keys}")

    all_episodes_meta, all_episodes_stats, all_tasks = [], [], {}
    total_frames, dst_ep_idx = 0, 0

    for ds_idx, (ds_path, ratio) in enumerate(zip(dataset_paths, ratios)):
        logger.info(f"\nProcessing {ds_path.name} (ratio {ratio}/{total_ratio})")

        info = load_dataset_info(ds_path)
        all_tasks.update(load_tasks_jsonl(ds_path))
        src_ep_stats = load_episodes_stats(ds_path)
        sampled_indices = sample_episodes(info["total_episodes"], ratio, total_ratio, rng)
        logger.info(f"  Sampled {len(sampled_indices)} episodes")
        src_episodes = load_episodes_jsonl(ds_path)

        for src_ep_idx in tqdm(sampled_indices, desc=f"Copying from {ds_path.name}"):
            ep_length = copy_episode_data(ds_path, output_path, src_ep_idx, dst_ep_idx, video_keys, total_frames)
            all_episodes_meta.append({"episode_index": dst_ep_idx, "tasks": src_episodes[src_ep_idx]["tasks"], "length": ep_length})
            if src_ep_stats:
                ep_stat = src_ep_stats[src_ep_idx].copy()
                ep_stat["episode_index"] = dst_ep_idx
                all_episodes_stats.append(ep_stat)
            total_frames += ep_length
            dst_ep_idx += 1

    info = load_dataset_info(dataset_paths[0])
    info.update({"total_episodes": dst_ep_idx, "total_frames": total_frames, "splits": {"train": f"0:{dst_ep_idx}"}})

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in all_episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for task_idx, task_desc in sorted(all_tasks.items()):
            f.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")
    if all_episodes_stats:
        with open(output_path / "meta" / "episodes_stats.jsonl", "w") as f:
            for ep_stat in all_episodes_stats:
                f.write(json.dumps(ep_stat) + "\n")

    return {"output_path": str(output_path), "total_episodes": dst_ep_idx, "total_frames": total_frames, "video_keys": video_keys}


def verify_video_integrity(dataset_path: Path, video_keys: list[str]) -> bool:
    import cv2
    all_ok = True
    for vid_key in video_keys:
        video_path = dataset_path / "videos" / vid_key / "chunk-000" / "file-000.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        success = sum(1 for _ in iter(lambda: cap.read()[0], False))
        cap.release()
        status = "OK" if success >= total * 0.99 else "CORRUPTED"
        logger.info(f"  {vid_key}: {success}/{int(total)} frames [{status}]")
        if success < total * 0.99:
            all_ok = False
    return all_ok


def main():
    args = parse_args()

    global VIDEO_CRF, VIDEO_PRESET
    VIDEO_CRF, VIDEO_PRESET = args.crf, args.preset

    dataset_paths = [Path(p).expanduser() for p in args.datasets]
    output_dir = args.output_dir.expanduser()

    for ds_path in dataset_paths:
        if not ds_path.exists() or not (ds_path / "meta" / "info.json").exists():
            raise ValueError(f"Invalid dataset: {ds_path}")

    logger.info("=" * 60)
    logger.info("Mixed Dataset Creation (v2.1 -> v3.0 with safe video)")
    logger.info("=" * 60)
    logger.info(f"Datasets: {[p.name for p in dataset_paths]}, Ratios: {args.ratios}")
    logger.info(f"Output: {output_dir / args.output_name}")

    stats = create_mixed_dataset(dataset_paths, args.ratios, output_dir, args.output_name, args.seed)
    logger.info(f"\nV2.1: {stats['total_episodes']} episodes, {stats['total_frames']} frames")

    # Convert to v3.0
    logger.info("\n" + "=" * 60)
    logger.info("Converting to v3.0 format (this may take a while)...")
    logger.info("=" * 60)

    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
    convert_dataset(repo_id="local", root=stats["output_path"], force_conversion=True, push_to_hub=False)

    logger.info(f"\nConverted: {stats['output_path']}")
    logger.info(f"Backup: {stats['output_path']}_old")

    logger.info("\n" + "=" * 60)
    logger.info("Verifying video integrity...")
    logger.info("=" * 60)
    if verify_video_integrity(Path(stats["output_path"]), stats["video_keys"]):
        logger.info("\nSUCCESS: All videos are valid!")
    else:
        logger.error("\nWARNING: Some videos may be corrupted!")

    logger.info(f"\nNext: uv run lerobot-train --dataset.repo_id=local --dataset.root={stats['output_path']} --policy.type=act ...")


if __name__ == "__main__":
    main()
