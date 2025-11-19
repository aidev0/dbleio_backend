# HPD Video Editing Script

This script combines segments from multiple source videos to create an optimized marketing video for Half Price Drapes.

## Prerequisites

1. **ffmpeg** - Already installed at `/opt/homebrew/bin/ffmpeg`
2. **Python virtual environment** - Located in `venv/`
3. **Python packages** - `ffmpeg-python` and `moviepy` (installed)

## Source Videos

Place your source video files in `data/videos/`:

```
backend/data/videos/
├── hpd_video_1.mov
├── hpd_video_2.mov
├── hpd_video_3.mov
└── hpd_video_4.mov
```

## Video Timeline

The script creates a combined video using the following timeline:

| Output Time | Source Video | Source Time | Duration |
|-------------|--------------|-------------|----------|
| 0:00 - 0:05 | hpd_video_4.mov | 0:00 - 0:05 | 5s |
| 0:05 - 0:07 | hpd_video_1.mov | 0:00 - 0:02 | 2s |
| 0:07 - 0:09 | hpd_video_2.mov | 0:01 - 0:03 | 2s |
| 0:09 - 0:11 | hpd_video_3.mov | 0:01 - 0:03 | 2s |
| 0:11 - 0:13 | hpd_video_2.mov | 0:10 - 0:12 | 2s |
| 0:13 - 0:15 | hpd_video_2.mov | 0:21 - 0:23 | 2s |
| 0:15 - 0:17 | hpd_video_2.mov | 0:19 - 0:21 | 2s |
| 0:17 - 0:18 | hpd_video_2.mov | 0:18 - 0:19 | 1s |
| 0:18 - 0:21 | hpd_video_4.mov | 0:13 - 0:16 | 3s |
| 0:21 - 0:24 | hpd_video_4.mov | 0:15 - 0:18 | 3s |

**Total Duration: ~24 seconds**

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the script
python create_combined_video.py
```

This will:
- Read source videos from `data/videos/`
- Create combined video at `data/videos/combined_marketing_video.mp4`

### Advanced Usage

```bash
# Custom input directory
python create_combined_video.py --input-dir /path/to/videos

# Custom output file
python create_combined_video.py --output /path/to/output.mp4

# Both custom
python create_combined_video.py --input-dir /path/to/videos --output /path/to/output.mp4
```

## Output

The script will generate:
- **File**: `data/videos/combined_marketing_video.mp4`
- **Codec**: H.264 (libx264)
- **Audio**: AAC
- **FPS**: 30
- **Bitrate**: 5000k
- **Quality**: Medium preset (good balance of speed and quality)

## Example Output

```
============================================================
HPD Marketing Video Creator
============================================================

Processing video segments...

✓ Segment  1: hpd_video_4.mov    | 0.0s - 5.0s   | Duration: 5.0s | Output: 0.0s - 5.0s
✓ Segment  2: hpd_video_1.mov    | 0.0s - 2.0s   | Duration: 2.0s | Output: 5.0s - 7.0s
✓ Segment  3: hpd_video_2.mov    | 1.0s - 3.0s   | Duration: 2.0s | Output: 7.0s - 9.0s
...

Total segments: 10
Combined duration: 24.0 seconds (0.4 minutes)

Combining video segments...
Exporting to: data/videos/combined_marketing_video.mp4

============================================================
✓ Video successfully created: /path/to/combined_marketing_video.mp4
============================================================
```

## Troubleshooting

### Missing Source Videos

If you see:
```
Error: Input directory 'data/videos' not found!
```

Create the directory and add your videos:
```bash
mkdir -p data/videos
# Copy your .mov files to data/videos/
```

### ffmpeg Errors

If you encounter ffmpeg errors, ensure ffmpeg is properly installed:
```bash
which ffmpeg
# Should output: /opt/homebrew/bin/ffmpeg
```

### Memory Issues

For large videos, you may need to:
1. Reduce the bitrate: Change `bitrate='5000k'` to `bitrate='3000k'`
2. Use a faster preset: Change `preset='medium'` to `preset='fast'`

## Editing the Timeline

To modify the video timeline, edit the `TIMELINE` list in `create_combined_video.py`:

```python
TIMELINE = [
    # (source_video, start_time, end_time)
    ("hpd_video_4.mov", 0, 5),      # 5 seconds from video 4
    ("hpd_video_1.mov", 0, 2),      # 2 seconds from video 1
    # Add more segments...
]
```

## Dependencies

All required packages are in `requirements.txt`:

```
ffmpeg-python
moviepy
```

To reinstall:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
