#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mix multiple LeRobot v2.1 datasets with specified ratios for training.

================================================================================
OVERVIEW
================================================================================

This script creates a mixed dataset by randomly sampling episodes from multiple
source datasets according to specified ratios. This is useful for:
  - Training on multiple tasks with different proportions
  - Data augmentation by mixing forward/reverse task demonstrations
  - Curriculum learning with controlled task distribution

Input:  Multiple v2.1 format datasets (from merge_datasets_v21.py)
Output: Mixed dataset ready for lerobot-train

================================================================================
USAGE
================================================================================

Basic usage (v2.1 output only):
    $ uv run python scripts/create_mixed_dataset_v21.py \
        --datasets /path/to/dataset1 /path/to/dataset2 \
        --ratios 9 1 \
        --output-dir /path/to/output \
        --output-name mixed_9_1

With v3.0 conversion (required for lerobot-train):
    $ uv run python scripts/create_mixed_dataset_v21.py \
        --datasets /path/to/dataset1 /path/to/dataset2 \
        --ratios 9 1 \
        --output-dir /path/to/output \
        --output-name mixed_9_1 \
        --convert-to-v30

================================================================================
ARGUMENTS
================================================================================

--datasets       Required. List of source dataset paths (v2.1 format).
                 Each path should point to a dataset directory containing
                 meta/, data/, and videos/ subdirectories.

--ratios         Required. Mixing ratios corresponding to each dataset.
                 Example: --ratios 9 1 means 90% from first dataset,
                 10% from second dataset.

--output-dir     Required. Directory where the mixed dataset will be created.

--output-name    Required. Name of the mixed dataset (subdirectory name).

--seed           Optional. Random seed for reproducible episode sampling.
                 Default: 42

--convert-to-v30 Optional. Convert output to v3.0 format after mixing.
                 Default: False (output remains in v2.1 format)

================================================================================
EXAMPLES
================================================================================

Example 1: Mix fwd and bwd-rev tasks at 9:1 ratio
    $ uv run python scripts/create_mixed_dataset_v21.py \
        --datasets \
            ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_fwd_1897 \
            ~/projects/TRaDA-data-real/merged_v21/Agilex_Cobot_Magic_insert_wedge_bwd_1910-rev \
        --ratios 9 1 \
        --output-dir ~/projects/TRaDA-data-real/mixed_v21 \
        --output-name mixed_fwd_bwdrev_9_1 \
        --seed 42 \
        --convert-to-v30

Example 2: Mix four datasets at 4:3:2:1 ratio
    $ uv run python scripts/create_mixed_dataset_v21.py \
        --datasets \
            /path/to/fwd \
            /path/to/bwd \
            /path/to/fwd-rev \
            /path/to/bwd-rev \
        --ratios 4 3 2 1 \
        --output-dir ~/projects/TRaDA-data-real/mixed_v21 \
        --output-name mixed_all_4_3_2_1 \
        --convert-to-v30

Example 3: Equal mix of two tasks (5:5)
    $ uv run python scripts/create_mixed_dataset_v21.py \
        --datasets /path/to/task_a /path/to/task_b \
        --ratios 5 5 \
        --output-dir ~/projects/TRaDA-data-real/mixed_v21 \
        --output-name mixed_equal \
        --convert-to-v30

================================================================================
TRAINING AFTER MIXING
================================================================================

After creating the mixed dataset, train with lerobot-train:

    $ uv run lerobot-train \
        --dataset.repo_id=local \
        --dataset.root=/path/to/mixed_dataset \
        --policy.type=act \
        --output_dir=outputs/train/act_mixed \
        --training.steps=50000 \
        --training.batch_size=8 \
        --device=cuda

================================================================================
SAMPLING ALGORITHM
================================================================================

For each dataset i with ratio r_i:
    episodes_to_sample_i = floor(total_episodes_i * r_i / sum(ratios))

Episodes are randomly sampled without replacement using the specified seed.
The resulting mixed dataset preserves all episode data, videos, and metadata.

================================================================================
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Mix multiple LeRobot v2.1 datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset paths (v2.1 format)",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=int,
        required=True,
        help="Mixing ratios (e.g., 9 1 for 9:1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for mixed dataset",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Name of the mixed dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--convert-to-v30",
        action="store_true",
        default=True,
        help="Convert to v3.0 format after mixing (default: True)",
    )
    return parser.parse_args()


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    """Load dataset info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episodes_jsonl(dataset_path: Path) -> list[dict]:
    """Load episodes.jsonl."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_episodes_stats(dataset_path: Path) -> list[dict]:
    """Load episodes_stats.jsonl."""
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return []
    stats = []
    with open(stats_path) as f:
        for line in f:
            stats.append(json.loads(line))
    return stats


def load_tasks_jsonl(dataset_path: Path) -> dict[int, str]:
    """Load tasks.jsonl and return task_index -> task mapping."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = {}
    with open(tasks_path) as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_index"]] = task["task"]
    return tasks


def get_video_keys(dataset_path: Path) -> list[str]:
    """Get video keys from dataset."""
    video_dir = dataset_path / "videos" / "chunk-000"
    if video_dir.exists():
        return [d.name for d in video_dir.iterdir() if d.is_dir()]
    return []


def sample_episodes(
    n_episodes: int,
    ratio: int,
    total_ratio: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample episode indices based on ratio."""
    n_samples = int(n_episodes * ratio / total_ratio)
    n_samples = max(1, n_samples)  # At least 1 episode
    all_indices = list(range(n_episodes))
    return sorted(rng.choice(all_indices, size=n_samples, replace=False).tolist())


def copy_episode_data(
    src_dataset: Path,
    dst_dataset: Path,
    src_ep_idx: int,
    dst_ep_idx: int,
    video_keys: list[str],
    total_frames_before: int,
) -> tuple[int, str]:
    """
    Copy episode data (parquet + videos) from source to destination.

    Returns:
        Tuple of (episode_length, task_description)
    """
    # Copy parquet file
    src_parquet = src_dataset / "data" / "chunk-000" / f"episode_{src_ep_idx:06d}.parquet"
    dst_parquet = dst_dataset / "data" / "chunk-000" / f"episode_{dst_ep_idx:06d}.parquet"

    # Read, modify, write parquet
    table = pq.read_table(src_parquet)
    df = table.to_pandas()
    episode_length = len(df)

    # Update episode_index
    df["episode_index"] = dst_ep_idx

    # Update index (global frame index)
    df["index"] = df.index + total_frames_before

    # Write modified parquet
    df.to_parquet(dst_parquet)

    # Copy video files
    for vid_key in video_keys:
        src_video = src_dataset / "videos" / "chunk-000" / vid_key / f"episode_{src_ep_idx:06d}.mp4"
        dst_video_dir = dst_dataset / "videos" / "chunk-000" / vid_key
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / f"episode_{dst_ep_idx:06d}.mp4"

        if src_video.exists():
            shutil.copy2(src_video, dst_video)

    # Get task description from first frame
    task_desc = df["task_index"].iloc[0] if "task_index" in df.columns else 0

    return episode_length, task_desc


def create_mixed_dataset(
    dataset_paths: list[Path],
    ratios: list[int],
    output_dir: Path,
    output_name: str,
    seed: int,
) -> dict[str, Any]:
    """
    Create mixed dataset from multiple source datasets.

    Returns:
        Dictionary with mixing statistics.
    """
    rng = np.random.default_rng(seed)

    # Validate inputs
    if len(dataset_paths) != len(ratios):
        raise ValueError(f"Number of datasets ({len(dataset_paths)}) must match number of ratios ({len(ratios)})")

    total_ratio = sum(ratios)

    # Create output directory structure
    output_path = output_dir / output_name
    if output_path.exists():
        logger.warning(f"Output path exists, removing: {output_path}")
        shutil.rmtree(output_path)

    output_data_dir = output_path / "data" / "chunk-000"
    output_video_dir = output_path / "videos" / "chunk-000"
    output_meta_dir = output_path / "meta"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    # Collect video keys from first dataset
    video_keys = get_video_keys(dataset_paths[0])
    logger.info(f"Video keys: {video_keys}")

    # Track statistics
    all_episodes_meta = []
    all_episodes_stats = []
    all_tasks = {}  # task_index -> task_description
    total_frames = 0
    dst_ep_idx = 0

    # Process each dataset
    for ds_idx, (ds_path, ratio) in enumerate(zip(dataset_paths, ratios)):
        logger.info(f"\nProcessing dataset {ds_idx + 1}/{len(dataset_paths)}: {ds_path.name}")
        logger.info(f"  Ratio: {ratio}/{total_ratio}")

        # Load dataset info
        info = load_dataset_info(ds_path)
        n_episodes = info["total_episodes"]

        # Load tasks
        tasks = load_tasks_jsonl(ds_path)
        all_tasks.update(tasks)

        # Load episode stats
        src_ep_stats = load_episodes_stats(ds_path)

        # Sample episodes
        sampled_indices = sample_episodes(n_episodes, ratio, total_ratio, rng)
        logger.info(f"  Sampled {len(sampled_indices)} episodes from {n_episodes} total")

        # Load source episodes metadata
        src_episodes = load_episodes_jsonl(ds_path)

        # Copy sampled episodes
        for src_ep_idx in sampled_indices:
            ep_length, task_idx = copy_episode_data(
                src_dataset=ds_path,
                dst_dataset=output_path,
                src_ep_idx=src_ep_idx,
                dst_ep_idx=dst_ep_idx,
                video_keys=video_keys,
                total_frames_before=total_frames,
            )

            # Build episode metadata
            src_ep_meta = src_episodes[src_ep_idx]
            ep_meta = {
                "episode_index": dst_ep_idx,
                "tasks": src_ep_meta["tasks"],
                "length": ep_length,
            }
            all_episodes_meta.append(ep_meta)

            # Copy episode stats if available
            if src_ep_stats:
                ep_stat = src_ep_stats[src_ep_idx].copy()
                ep_stat["episode_index"] = dst_ep_idx
                all_episodes_stats.append(ep_stat)

            total_frames += ep_length
            dst_ep_idx += 1

    logger.info(f"\nTotal episodes: {dst_ep_idx}")
    logger.info(f"Total frames: {total_frames}")

    # Write meta/info.json
    # Use first dataset's info as template
    info = load_dataset_info(dataset_paths[0])
    info["total_episodes"] = dst_ep_idx
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{dst_ep_idx}"}

    info_path = output_meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Wrote: {info_path}")

    # Write meta/episodes.jsonl
    episodes_path = output_meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep_meta in all_episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")
    logger.info(f"Wrote: {episodes_path}")

    # Write meta/tasks.jsonl
    tasks_path = output_meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for task_idx, task_desc in sorted(all_tasks.items()):
            f.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")
    logger.info(f"Wrote: {tasks_path}")

    # Write meta/episodes_stats.jsonl if available
    if all_episodes_stats:
        stats_path = output_meta_dir / "episodes_stats.jsonl"
        with open(stats_path, "w") as f:
            for ep_stat in all_episodes_stats:
                f.write(json.dumps(ep_stat) + "\n")
        logger.info(f"Wrote: {stats_path}")

    # Note: stats.json will be recomputed by lerobot when loading

    return {
        "output_path": str(output_path),
        "total_episodes": dst_ep_idx,
        "total_frames": total_frames,
        "video_keys": video_keys,
    }


def main():
    args = parse_args()

    # Convert to Path objects
    dataset_paths = [Path(p).expanduser() for p in args.datasets]
    output_dir = args.output_dir.expanduser()

    # Validate datasets exist
    for ds_path in dataset_paths:
        if not ds_path.exists():
            raise ValueError(f"Dataset not found: {ds_path}")
        if not (ds_path / "meta" / "info.json").exists():
            raise ValueError(f"Not a valid v2.1 dataset: {ds_path}")

    logger.info("=" * 60)
    logger.info("Mixed Dataset Creation (v2.1 format)")
    logger.info("=" * 60)
    logger.info(f"Datasets: {[p.name for p in dataset_paths]}")
    logger.info(f"Ratios: {args.ratios}")
    logger.info(f"Output: {output_dir / args.output_name}")
    logger.info(f"Seed: {args.seed}")

    stats = create_mixed_dataset(
        dataset_paths=dataset_paths,
        ratios=args.ratios,
        output_dir=output_dir,
        output_name=args.output_name,
        seed=args.seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Output: {stats['output_path']}")
    logger.info(f"Episodes: {stats['total_episodes']}")
    logger.info(f"Frames: {stats['total_frames']}")
    logger.info(f"Video keys: {stats['video_keys']}")

    # Convert to v3.0 if requested
    if args.convert_to_v30:
        logger.info("\n" + "=" * 60)
        logger.info("Converting to v3.0 format...")
        logger.info("=" * 60)
        from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
        convert_dataset(
            repo_id="local",
            root=stats["output_path"],
            force_conversion=True,
            push_to_hub=False,
        )
        # Note: convert_dataset converts in-place, backing up v2.1 to _old
        logger.info(f"\nConverted to v3.0 in-place: {stats['output_path']}")
        logger.info(f"Original v2.1 backed up to: {stats['output_path']}_old")

    logger.info("\nNext step: Train with lerobot-train")
    logger.info(f"  uv run lerobot-train --dataset.repo_id=local --dataset.root={stats['output_path']} --policy.type=act ...")


if __name__ == "__main__":
    main()
