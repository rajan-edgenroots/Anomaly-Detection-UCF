"""
Utility functions for video input validation, Direct Video URL handling,
file uploads, video metadata extraction, and temporary file management.
"""

import os
import re
import time
import shutil
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import requests
from config import CFG


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass


class InvalidURLError(VideoProcessingError):
    """Exception raised when an invalid or disallowed URL is provided."""
    pass


class InvalidFileError(VideoProcessingError):
    """Exception raised when an invalid file extension or oversized file is provided."""
    pass


class CorruptedVideoError(VideoProcessingError):
    """Exception raised when a video file cannot be opened or read."""
    pass


class DownloadError(VideoProcessingError):
    """Exception raised when a video download fails."""
    pass


def validate_file_extension(filename: str) -> str:
    """
    Validate that the file extension is allowed (.mp4, .avi, .mov, .mkv).

    Args:
        filename: Name of the uploaded file.

    Returns:
        Cleaned lowercase file extension.
    
    Raises:
        InvalidFileError: If file extension is not permitted.
    """
    if "." not in filename:
        raise InvalidFileError("File name has no extension.")
    
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in CFG.allowed_extensions:
        allowed = ", ".join(f".{e}" for e in sorted(CFG.allowed_extensions))
        raise InvalidFileError(f"Unsupported file format '.{ext}'. Allowed formats: {allowed}")
    return ext


def validate_video_url(url: str) -> str:
    """
    Validate direct video URLs, cloud storage links (e.g. Google Cloud Storage, GCS signed URLs, CDN links).

    Args:
        url: The candidate URL string.

    Returns:
        Sanitized URL string.

    Raises:
        InvalidURLError: If URL structure or protocol is invalid.
    """
    url = url.strip()
    if not url:
        raise InvalidURLError("Video URL cannot be empty.")

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise InvalidURLError(f"Invalid URL structure: {exc}")

    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError("URL must use HTTP or HTTPS protocol.")

    if not parsed.netloc:
        raise InvalidURLError("URL must contain a valid domain or host name.")

    return url


def download_video_from_url(url: str, output_dir: Path) -> Path:
    """
    Download video asset from direct URL or Cloud Storage link (GCS, S3, CDN, signed URLs).

    Args:
        url: Validated URL string.
        output_dir: Local path to store downloaded video file.

    Returns:
        Path to the downloaded video file.

    Raises:
        DownloadError: If download fails or video file is empty.
    """
    validated_url = validate_video_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_path = urllib.parse.urlparse(validated_url).path
    filename = Path(parsed_path).name

    # Check if filename extracted from path has valid video extension
    has_valid_ext = False
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in CFG.allowed_extensions:
            has_valid_ext = True

    if not has_valid_ext:
        filename = f"video_{int(time.time())}.mp4"

    target_path = output_dir / filename

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(validated_url, stream=True, headers=headers, timeout=120)
        response.raise_for_status()

        # Try to parse filename from Content-Disposition header if available
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            match = re.search(r'filename=["\']?([^"\';]+)["\']?', content_disposition)
            if match:
                cd_filename = match.group(1).strip()
                if "." in cd_filename and cd_filename.rsplit(".", 1)[1].lower() in CFG.allowed_extensions:
                    target_path = output_dir / cd_filename

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        if not target_path.exists() or target_path.stat().st_size == 0:
            raise DownloadError("Downloaded video file is empty or corrupted.")

        return target_path

    except requests.exceptions.RequestException as exc:
        raise DownloadError(f"Network error downloading video from URL: {exc}")
    except Exception as exc:
        raise DownloadError(f"Failed to process video from URL: {exc}")


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extract frame rate, total frame count, resolution, and duration from video file.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary containing video parameters.

    Raises:
        CorruptedVideoError: If OpenCV fails to open the video.
    """
    if not video_path.exists():
        raise CorruptedVideoError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CorruptedVideoError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not (fps > 0):
        fps = 30.0  # Reasonable default fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    cap.release()

    return {
        "fps": float(fps),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(float(duration_sec), 2),
    }


def prune_old_uploads(upload_dir: Path, max_age_seconds: int = 3600) -> None:
    """
    Prune files in upload directory older than max_age_seconds to prevent disk bloat.

    Args:
        upload_dir: Directory containing uploaded/downloaded video files.
        max_age_seconds: Threshold age in seconds (default 1 hour).
    """
    if not upload_dir.exists():
        return
    now = time.time()
    for file_path in upload_dir.glob("*"):
        if file_path.is_file():
            try:
                file_age = now - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
            except Exception:
                pass
