#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge scattered episode directories into unified LeRobot datasets (v2.1 format).

================================================================================
OVERVIEW
================================================================================

This script merges individual episode directories (each containing one episode's
data and videos) into a unified LeRobot dataset in v2.1 format. Optionally, it
can convert the output to v3.0 format for use with the latest lerobot training.

Input:  Scattered episode directories (one episode per directory)
Output: Unified dataset ready for lerobot-train

================================================================================
USAGE
================================================================================

Basic usage (v2.1 output only):
    $ cd /path/to/lerobot
    $ uv run python scripts/merge_datasets_v21.py

With v3.0 conversion:
    $ uv run python scripts/merge_datasets_v21.py --convert-to-v30

================================================================================
CONFIGURATION
================================================================================

Edit DATASET_CONFIGS at the bottom of this file to specify:
  - source_dir: Path to scattered episode directories
  - output_name: Name for the merged dataset
  - task_description: Natural language task description
  - episode_prefix: Prefix to identify episode directories

Example configuration:
    DATASET_CONFIGS = {
        "my_task": {
            "source_dir": Path.home() / "data/scattered_episodes",
            "output_name": "my_robot_task",
            "task_description": "Pick up object and place in container.",
            "episode_prefix": "episode_",
        },
    }

================================================================================
OUTPUT STRUCTURE
================================================================================

v2.1 format (default):
    <output_dir>/
    ├── meta/
    │   ├── info.json           # Dataset metadata (codebase_version: "v2.1")
    │   ├── episodes.jsonl      # Per-episode metadata
    │   ├── tasks.jsonl         # Task descriptions
    │   ├── episodes_stats.jsonl # Per-episode statistics
    │   └── stats.json          # Aggregated statistics
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet  # One file per episode
    │       ├── episode_000001.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.image_top/
            │   ├── episode_000000.mp4  # One video per episode
            │   └── ...
            └── ...

v3.0 format (with --convert-to-v30):
    <output_dir>/
    ├── meta/
    │   ├── info.json           # Dataset metadata (codebase_version: "v3.0")
    │   ├── tasks.parquet       # Task descriptions in parquet
    │   ├── stats.json
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet  # Episode metadata
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet  # All episodes aggregated
    └── videos/
        ├── observation.images.image_top/
        │   └── chunk-000/
        │       └── file-000.mp4  # All episode videos concatenated
        └── ...

================================================================================
EXAMPLE
================================================================================

# Step 1: Configure and run merge
$ cd /home/rogerspyke/projects/TRaDA/third_party/lerobot
$ uv run python scripts/merge_datasets_v21.py --convert-to-v30

# Step 2: Train on the merged dataset
$ uv run lerobot-train \
    --dataset.repo_id=local \
    --dataset.root=/home/rogerspyke/projects/TRaDA-data-real/merged_v21/<dataset_name> \
    --policy.type=act \
    --output_dir=outputs/train/act_<dataset_name>

================================================================================
ARGUMENTS
================================================================================

--convert-to-v30    Convert output to v3.0 format after merging.
                   Default: False (output remains in v2.1 format)

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
from tqdm import tqdm

# Import lerobot's stats computation functions
from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.utils.utils import unflatten_dict

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Natural language task descriptions
TASK_DESCRIPTIONS = {
    "bwd": "Pick up the red triangular wedge from the white square mold and place it on the table.",
    "fwd": "Pick up the red triangular wedge from the table and insert it into the white square mold.",
}

DATASET_CONFIGS = {
    "bwd": {
        "source_dir": Path.home() / "projects/TRaDA-data-real/scattered/Agilex_Cobot_Magic_insert_wedge_bwd_1910",
        "output_name": "Agilex_Cobot_Magic_insert_wedge_bwd_1910",
        "task_description": TASK_DESCRIPTIONS["bwd"],
        "episode_prefix": "Agilex_Cobot_Magic_insert_wedge_bwd_1910_",
    },
    "fwd": {
        "source_dir": Path.home() / "projects/TRaDA-data-real/scattered/Agilex_Cobot_Magic_insert_wedge_fwd_1897",
        "output_name": "Agilex_Cobot_Magic_insert_wedge_fwd_1897",
        "task_description": TASK_DESCRIPTIONS["fwd"],
        "episode_prefix": "Agilex_Cobot_Magic_insert_wedge_fwd_1897_",
    },
}

# v2.1 format constants
V21 = "v2.1"


def get_episode_dirs(source_dir: Path, prefix: str) -> list[Path]:
    """Get sorted list of episode directories."""
    episode_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ])
    return episode_dirs


def prepare_episode_data_for_stats(df: pd.DataFrame, features: dict) -> dict:
    """
    Prepare episode data for stats computation.

    Convert DataFrame columns to numpy arrays suitable for compute_episode_stats.

    @input:
        - df: DataFrame containing episode data
        - features: Feature definitions from info.json

    @output:
        - Dictionary mapping feature names to numpy arrays
    """
    episode_data = {}
    for key, ft in features.items():
        if ft["dtype"] == "string":
            continue

        if key not in df.columns:
            continue

        # Skip video/image features (stored as files, not in parquet)
        if ft["dtype"] in ["video", "image"]:
            continue

        data = df[key].values

        # Handle object-dtype columns containing numpy arrays (e.g. action, observation.state)
        if data.dtype == object:
            try:
                data = np.stack(data)
            except (ValueError, TypeError):
                continue

        episode_data[key] = data

    return episode_data


def stats_to_json_serializable(stats: dict) -> dict:
    """
    Convert numpy arrays in stats to JSON-serializable lists.

    @input:
        - stats: Dictionary with numpy array values

    @output:
        - Dictionary with list values suitable for JSON serialization
    """
    result = {}
    for feature_key, feature_stats in stats.items():
        result[feature_key] = {}
        for stat_key, stat_value in feature_stats.items():
            if isinstance(stat_value, np.ndarray):
                result[feature_key][stat_key] = stat_value.tolist()
            else:
                result[feature_key][stat_key] = stat_value
    return result


def merge_episodes(
    source_dir: Path,
    output_dir: Path,
    episode_prefix: str,
    task_description: str,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Merge all episode directories into a single dataset in v2.1 format.

    Returns:
        Dictionary with dataset statistics.
    """
    episode_dirs = get_episode_dirs(source_dir, episode_prefix)
    logger.info(f"Found {len(episode_dirs)} episodes for {dataset_name}")

    if len(episode_dirs) == 0:
        raise ValueError(f"No episode directories found in {source_dir}")

    # Create output directory structure (v2.1 format)
    output_data_dir = output_dir / "data" / "chunk-000"
    output_video_dir = output_dir / "videos" / "chunk-000"
    output_meta_dir = output_dir / "meta"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data and video keys
    video_keys = []
    total_frames = 0

    # Episode metadata for v2.1 format (JSONL)
    episodes_metadata = []
    episode_stats_list = []

    # Read source info.json once (from first episode)
    first_ep_dir = episode_dirs[0]
    first_info_path = first_ep_dir / "meta" / "info.json"
    with open(first_info_path) as f:
        source_info = json.load(f)
    features = source_info.get("features", {})

    # Filter features for stats computation (exclude video/image)
    non_video_features = {
        k: v for k, v in features.items()
        if v.get("dtype") not in ["video", "image", "string"]
    }

    for ep_idx, ep_dir in enumerate(tqdm(episode_dirs, desc=f"Merging {dataset_name}")):
        # Read episode data
        ep_data_dir = ep_dir / "data" / "chunk-000"
        ep_parquet = ep_data_dir / "episode_000000.parquet"

        if not ep_parquet.exists():
            logger.warning(f"Parquet not found: {ep_parquet}")
            continue

        # Read parquet data
        table = pq.read_table(ep_parquet)
        df = table.to_pandas()
        ep_length = len(df)

        # Update episode_index to global index
        df["episode_index"] = ep_idx

        # Update index to global frame index
        df["index"] = df.index + total_frames

        # Write per-episode parquet file (v2.1 format: one file per episode)
        output_parquet = output_data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(output_parquet, index=False)

        # Get video keys from first episode
        if ep_idx == 0:
            ep_video_dir = ep_dir / "videos" / "chunk-000"
            if ep_video_dir.exists():
                video_keys = [d.name for d in ep_video_dir.iterdir() if d.is_dir()]
            logger.info(f"Video keys: {video_keys}")

        # Copy video files (v2.1 format: one video per episode)
        ep_video_dir = ep_dir / "videos" / "chunk-000"
        for vid_key in video_keys:
            vid_src_dir = ep_video_dir / vid_key
            if vid_src_dir.exists():
                vid_dst_dir = output_video_dir / vid_key
                vid_dst_dir.mkdir(parents=True, exist_ok=True)

                src_video = vid_src_dir / "episode_000000.mp4"
                dst_video = vid_dst_dir / f"episode_{ep_idx:06d}.mp4"

                if src_video.exists():
                    shutil.copy2(src_video, dst_video)

        # Build episode metadata for v2.1 JSONL format
        episode_meta = {
            "episode_index": ep_idx,
            "tasks": [task_description],
            "length": ep_length,
        }
        episodes_metadata.append(episode_meta)

        # Compute per-episode stats using lerobot's function
        episode_data = prepare_episode_data_for_stats(df, non_video_features)
        ep_stats = compute_episode_stats(episode_data, non_video_features)
        episode_stats_list.append({"episode_index": ep_idx, "stats": ep_stats})

        total_frames += ep_length

    # Write meta/info.json (v2.1 format)
    info = source_info.copy()

    # Update info for v2.1
    info["codebase_version"] = V21
    info["total_episodes"] = len(episode_dirs)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(episode_dirs)}"}

    with open(output_meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Wrote info.json with codebase_version={V21}")

    # Write meta/tasks.jsonl (v2.1 format)
    tasks_jsonl_path = output_meta_dir / "tasks.jsonl"
    with open(tasks_jsonl_path, "w") as f:
        task_entry = {"task_index": 0, "task": task_description}
        f.write(json.dumps(task_entry) + "\n")
    logger.info(f"Wrote tasks.jsonl with task: {task_description}")

    # Write meta/episodes.jsonl (v2.1 format)
    episodes_jsonl_path = output_meta_dir / "episodes.jsonl"
    with open(episodes_jsonl_path, "w") as f:
        for ep_meta in episodes_metadata:
            f.write(json.dumps(ep_meta) + "\n")
    logger.info(f"Wrote episodes.jsonl with {len(episodes_metadata)} episodes")

    # Write meta/episodes_stats.jsonl (v2.1 format)
    # Convert numpy arrays to lists for JSON serialization
    episodes_stats_jsonl_path = output_meta_dir / "episodes_stats.jsonl"
    with open(episodes_stats_jsonl_path, "w") as f:
        for ep_stat_entry in episode_stats_list:
            json_stats = stats_to_json_serializable(ep_stat_entry["stats"])
            f.write(json.dumps({"episode_index": ep_stat_entry["episode_index"], "stats": json_stats}) + "\n")
    logger.info(f"Wrote episodes_stats.jsonl with {len(episode_stats_list)} entries")

    # Write meta/stats.json (aggregated stats)
    aggregated_stats = aggregate_stats([es["stats"] for es in episode_stats_list])
    aggregated_stats_json = stats_to_json_serializable(aggregated_stats)
    stats_json_path = output_meta_dir / "stats.json"
    with open(stats_json_path, "w") as f:
        json.dump(aggregated_stats_json, f, indent=4)
    logger.info("Wrote stats.json with aggregated statistics")

    return {
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "video_keys": video_keys,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge scattered episode directories into unified LeRobot datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--convert-to-v30",
        action="store_true",
        help="Convert output to v3.0 format after merging (default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_base = Path.home() / "projects/TRaDA-data-real/merged_v21"
    output_base.mkdir(parents=True, exist_ok=True)

    results = {}

    for dataset_type, config in DATASET_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_type} dataset")
        logger.info(f"{'='*60}")

        output_dir = output_base / config["output_name"]

        # Clean existing output
        if output_dir.exists():
            shutil.rmtree(output_dir)

        stats = merge_episodes(
            source_dir=config["source_dir"],
            output_dir=output_dir,
            episode_prefix=config["episode_prefix"],
            task_description=config["task_description"],
            dataset_name=config["output_name"],
        )

        results[dataset_type] = {
            "output_dir": str(output_dir),
            **stats,
        }

        logger.info(f"\n{dataset_type} dataset merged successfully:")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Episodes: {stats['total_episodes']}")
        logger.info(f"  Frames: {stats['total_frames']}")
        logger.info(f"  Task: {config['task_description']}")

        # Convert to v3.0 if requested
        if args.convert_to_v30:
            logger.info(f"\n{'='*60}")
            logger.info(f"Converting {config['output_name']} to v3.0 format...")
            logger.info(f"{'='*60}")
            from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
            convert_dataset(
                repo_id=config["output_name"],
                root=str(output_dir),
                force_conversion=True,
                push_to_hub=False,
            )
            logger.info(f"Converted to v3.0 in-place: {output_dir}")
            logger.info(f"Original v2.1 backed up to: {output_dir}_old")

    logger.info(f"\n{'='*60}")
    logger.info("Summary")
    logger.info(f"{'='*60}")
    for ds_type, res in results.items():
        logger.info(f"\n{ds_type.upper()}:")
        logger.info(f"  Output: {res['output_dir']}")
        logger.info(f"  Episodes: {res['total_episodes']}")
        logger.info(f"  Frames: {res['total_frames']}")

    logger.info(f"\n{'='*60}")
    logger.info("Next step: Train with lerobot-train")
    logger.info("  uv run lerobot-train \\")
    logger.info("    --dataset.repo_id=local \\")
    logger.info("    --dataset.root=<output_dir> \\")
    logger.info("    --policy.type=act \\")
    logger.info("    --output_dir=outputs/train/act_<dataset_name>")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()