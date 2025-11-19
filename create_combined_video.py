#!/usr/bin/env python3
"""
Video Editing Script for HPD Marketing Video
Combines segments from 4 source videos into a single optimized video
"""

from moviepy import VideoFileClip, concatenate_videoclips
import os

# Video segment timeline
# Format: (source_video, start_time, end_time, output_position)
TIMELINE = [
    # Segment 1: 0:00 - 0:05 from video 4
    ("hpd_video_4.mov", 0, 5),

    # Segment 2: 0:00 - 0:02 from video 1
    ("hpd_video_1.mov", 0, 2),

    # Segment 3: 0:01 - 0:03 from video 2
    ("hpd_video_2.mov", 1, 3),

    # Segment 4: 0:01 - 0:03 from video 3
    ("hpd_video_3.mov", 1, 3),

    # Segment 5: 0:10 - 0:12 from video 2
    ("hpd_video_2.mov", 10, 12),

    # Segment 6: 0:21 - 0:23 from video 2
    ("hpd_video_2.mov", 21, 23),

    # Segment 7: 0:19 - 0:21 from video 2
    ("hpd_video_2.mov", 19, 21),

    # Segment 8: 0:18 - 0:19 from video 2
    ("hpd_video_2.mov", 18, 19),

    # Segment 9: 0:13 - 0:16 from video 4 (adjusted to 3 seconds)
    ("hpd_video_4.mov", 13, 16),

    # Segment 10: 0:15 - 0:18 from video 4
    ("hpd_video_4.mov", 15, 18),
]

def create_combined_video(input_dir="data/videos", output_file="data/videos/combined_marketing_video.mp4"):
    """
    Create a combined video from multiple source videos based on the timeline

    Args:
        input_dir: Directory containing source video files
        output_file: Path for the output combined video
    """

    print("=" * 60)
    print("HPD Marketing Video Creator")
    print("=" * 60)
    print()

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        print(f"Please place your source videos in: {os.path.abspath(input_dir)}")
        print("\nExpected files:")
        print("  - hpd_video_1.mov")
        print("  - hpd_video_2.mov")
        print("  - hpd_video_3.mov")
        print("  - hpd_video_4.mov")
        return

    # Collect video clips
    clips = []
    total_duration = 0

    print("Processing video segments...")
    print()

    for i, (source_video, start_time, end_time) in enumerate(TIMELINE, 1):
        video_path = os.path.join(input_dir, source_video)

        # Check if source video exists
        if not os.path.exists(video_path):
            print(f"Warning: {source_video} not found at {video_path}")
            continue

        # Load and cut the video segment
        try:
            clip = VideoFileClip(video_path).subclipped(start_time, end_time)
            duration = end_time - start_time
            clips.append(clip)

            print(f"✓ Segment {i:2d}: {source_video:18s} | {start_time:5.1f}s - {end_time:5.1f}s | Duration: {duration:.1f}s | Output: {total_duration:.1f}s - {total_duration + duration:.1f}s")

            total_duration += duration

        except Exception as e:
            print(f"✗ Error processing {source_video} ({start_time}s-{end_time}s): {e}")

    if not clips:
        print("\nError: No video segments could be loaded!")
        print("Please ensure all source videos exist in the input directory.")
        return

    print()
    print(f"Total segments: {len(clips)}")
    print(f"Combined duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print()

    # Concatenate all clips
    print("Combining video segments...")
    final_clip = concatenate_videoclips(clips, method="compose")

    # Extract audio from video 2 and apply to combined video
    print("Adding background music from Video 2...")
    video2_path = os.path.join(input_dir, "hpd_video_2.mov")

    if os.path.exists(video2_path):
        try:
            # Load video 2 to get its audio
            video2 = VideoFileClip(video2_path)

            if video2.audio is not None:
                # Get the audio from video 2 - use only the first 24 seconds
                # Extract to a separate AudioFileClip to avoid reader issues
                from moviepy import AudioFileClip
                import tempfile

                # Create temporary audio file
                temp_audio = tempfile.mktemp(suffix='.m4a')
                print(f"  Extracting audio to temporary file...")
                video2.audio.write_audiofile(temp_audio, fps=44100, nbytes=2, codec='aac', logger=None)

                # Load the audio back as a fresh AudioFileClip
                background_audio = AudioFileClip(temp_audio)
                target_duration = final_clip.duration

                # Set the duration to match the video
                if background_audio.duration > target_duration:
                    print(f"  Trimming audio (original: {background_audio.duration:.1f}s, target: {target_duration:.1f}s)")
                    background_audio = background_audio.subclipped(0, target_duration)

                # Set the audio to the final clip
                final_clip = final_clip.with_audio(background_audio)
                print(f"✓ Audio from Video 2 added successfully (duration: {target_duration:.1f}s)")
            else:
                print("⚠ Video 2 has no audio track")

            video2.close()
        except Exception as e:
            print(f"⚠ Could not add audio from Video 2: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ Video 2 not found at {video2_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the final video
    print(f"Exporting to: {output_file}")
    final_clip.write_videofile(
        output_file,
        codec='libx264',
        audio_codec='aac',
        fps=30,
        preset='medium',
        bitrate='5000k'
    )

    # Clean up
    for clip in clips:
        clip.close()
    final_clip.close()

    print()
    print("=" * 60)
    print(f"✓ Video successfully created: {os.path.abspath(output_file)}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create combined HPD marketing video")
    parser.add_argument(
        "--input-dir",
        default="data/videos",
        help="Directory containing source videos (default: data/videos)"
    )
    parser.add_argument(
        "--output",
        default="data/videos/combined_marketing_video.mp4",
        help="Output file path (default: data/videos/combined_marketing_video.mp4)"
    )

    args = parser.parse_args()

    create_combined_video(args.input_dir, args.output)
