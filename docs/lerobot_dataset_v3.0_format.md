# LeRobot Dataset v3.0 Format Specification

## Overview

LeRobot Dataset v3.0 is a file-based, chunked storage format designed for efficient storage and streaming of robot learning data. This document provides the complete format specification based on the official LeRobot codebase.

## Codebase Version

```
CODEBASE_VERSION = "v3.0"
```

## Directory Structure

```
dataset_root/
├── meta/
│   ├── info.json              # Dataset schema and configuration
│   ├── stats.json             # Feature statistics (optional)
│   ├── tasks.parquet          # Task descriptions
│   └── episodes/
│       └── chunk-{chunk_index:03d}/
│           └── file-{file_index:03d}.parquet  # Episode metadata
├── data/
│   └── chunk-{chunk_index:03d}/
│       └── file-{file_index:03d}.parquet      # Frame data
└── videos/                    # (if video features exist)
    └── chunk-{chunk_index:03d}/
        └── {video_key}/
            └── episode_{episode_index:06d}.mp4  # Video files
```

**Note:** The video path template in `info.json` determines the actual structure:
- Template: `videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4`

## info.json Specification

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `codebase_version` | string | Must be `"v3.0"` |
| `robot_type` | string \| null | Robot type identifier |
| `total_episodes` | int | Total number of episodes |
| `total_frames` | int | Total number of frames |
| `total_tasks` | int | Number of unique tasks |
| `chunks_size` | int | Max files per chunk (default: 1000) |
| `fps` | int | Frames per second |
| `splits` | dict | Train/val splits, e.g., `{"train": "0:100"}` |
| `data_path` | string | Template for data parquet paths |
| `features` | dict | Feature definitions |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `data_files_size_in_mb` | int | Max parquet file size (default: 100) |
| `video_files_size_in_mb` | int | Max video file size (default: 200) |
| `video_path` | string \| null | Template for video paths |
| `image_path` | string \| null | Template for image paths |
| `audio_path` | string \| null | Template for audio paths |
| `total_videos` | int | Number of video streams |
| `total_chunks` | int | Number of chunks |

### Path Templates

```json
{
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"
}
```

### Feature Definition Schema

```json
{
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": ["joint1", "joint2", ...]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [26],
      "names": [...]
    },
    "observation.images.camera_name": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channels"],
      "info": {
        "video.fps": 30,
        "video.height": 480,
        "video.width": 640,
        "video.channels": 3,
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": null},
    "frame_index": {"dtype": "int64", "shape": [1], "names": null},
    "episode_index": {"dtype": "int64", "shape": [1], "names": null},
    "index": {"dtype": "int64", "shape": [1], "names": null},
    "task_index": {"dtype": "int64", "shape": [1], "names": null}
  }
}
```

### DEFAULT_FEATURES (Auto-populated)

These features are automatically populated by the recording pipeline:

```python
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
```

## tasks.parquet Specification

| Column | Type | Description |
|--------|------|-------------|
| `task` (index) | string | Natural language task description |
| `task_index` | int64 | Integer task ID |

Example:
```
task                                          | task_index
----------------------------------------------|-----------
"Pick up the red cube and place it in box"   | 0
```

## Episode Metadata (meta/episodes/*.parquet)

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `episode_index` | int64 | Episode ID |
| `tasks` | list[str] | Task descriptions for this episode |
| `length` | int64 | Number of frames in episode |
| `dataset_from_index` | int64 | Start frame index in dataset |
| `dataset_to_index` | int64 | End frame index in dataset |
| `data/chunk_index` | int64 | Chunk index for data parquet |
| `data/file_index` | int64 | File index for data parquet |

### Video Columns (per video key)

| Column | Type | Description |
|--------|------|-------------|
| `videos/{video_key}/chunk_index` | int64 | Chunk index for video |
| `videos/{video_key}/file_index` | int64 | File index for video |
| `videos/{video_key}/from_timestamp` | float64 | Start timestamp in video file |
| `videos/{video_key}/to_timestamp` | float64 | End timestamp in video file |

## Data Parquet Specification

### Required Columns (DEFAULT_FEATURES)

| Column | dtype | Description |
|--------|-------|-------------|
| `timestamp` | float32 | Time in seconds |
| `frame_index` | int64 | Frame index within episode |
| `episode_index` | int64 | Episode ID |
| `index` | int64 | Global frame index |
| `task_index` | int64 | Task ID |

### User-defined Features

- `action`: Robot action vector
- `observation.state`: Robot state vector
- Video features are **NOT** stored in data parquet (stored as MP4 files)

## Video Storage

- Format: MP4 (H.264/HEVC/AV1)
- Each episode's frames are concatenated into a single video per camera
- Videos are sharded by chunk/file indices
- Frame extraction uses timestamp offsets from episode metadata

## Example: Valid v3.0 Dataset

```
Agilex_Cobot_Magic_insert_wedge_bwd_1910/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.image_top/
        │   ├── episode_000000.mp4
        │   ├── episode_000001.mp4
        │   └── ...
        ├── observation.images.image_right/
        │   └── ...
        └── observation.images.image_left/
            └── ...
```

### info.json Example

```json
{
    "codebase_version": "v3.0",
    "robot_type": null,
    "total_episodes": 100,
    "total_frames": 42313,
    "total_tasks": 1,
    "total_videos": 3,
    "total_chunks": 1,
    "chunks_size": 10000,
    "fps": 30,
    "splits": {"train": "0:100"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
    "features": {
        "action": {"dtype": "float32", "shape": [14], "names": [...]},
        "observation.state": {"dtype": "float32", "shape": [26], "names": [...]},
        "observation.images.image_top": {"dtype": "video", "shape": [480, 640, 3], ...},
        "observation.images.image_right": {"dtype": "video", "shape": [480, 640, 3], ...},
        "observation.images.image_left": {"dtype": "video", "shape": [480, 640, 3], ...},
        "timestamp": {"dtype": "float32", "shape": [1], "names": null},
        "frame_index": {"dtype": "int64", "shape": [1], "names": null},
        "episode_index": {"dtype": "int64", "shape": [1], "names": null},
        "index": {"dtype": "int64", "shape": [1], "names": null},
        "task_index": {"dtype": "int64", "shape": [1], "names": null}
    }
}
```

## Validation Checklist

1. **codebase_version**: Must be exactly `"v3.0"`
2. **Directory structure**: `meta/`, `data/`, `videos/` directories exist
3. **info.json**: All required fields present
4. **tasks.parquet**: Has `task` index and `task_index` column
5. **Episode metadata**: All required columns present
6. **Data parquet**: DEFAULT_FEATURES columns with correct dtypes
7. **Video files**: Exist at paths specified by episode metadata
8. **Cross-validation**: `total_episodes`, `total_frames`, `total_tasks` match actual data

## Key Differences from v2.1

| Aspect | v2.1 | v3.0 |
|--------|------|------|
| File organization | One file per episode | Many episodes per file |
| Episode lookup | By filename | By metadata offsets |
| Video storage | Per-episode MP4 | Concatenated MP4 with offsets |
| Metadata | JSONL files | Chunked Parquet |
| Streaming | Not supported | Native Hub streaming |

## References

- Source code: `lerobot/datasets/dataset_metadata.py`
- Constants: `lerobot/utils/constants.py`
- Writer: `lerobot/datasets/dataset_writer.py`
- I/O utils: `lerobot/datasets/io_utils.py`
