"""
Inference pipeline for Federated Video Anomaly Detection.
Replicates video frame extraction, sliding window sequence generation,
batch prediction, probability aggregation, top-3 class calculation, and timeline generation with frame thumbnails.
"""

import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from config import CFG
from model import ModelLoader
from utils import CorruptedVideoError, VideoProcessingError, get_video_metadata

logger = logging.getLogger(__name__)


def frame_to_base64_jpeg(frame_norm: np.ndarray) -> str:
    """
    Convert float32 normalized RGB frame array [H, W, C] to base64 JPEG data URI string.

    Args:
        frame_norm: Normalized frame array with values in [0.0, 1.0].

    Returns:
        Base64 encoded JPEG image URI string.
    """
    try:
        frame_uint8 = (np.clip(frame_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as exc:
        logger.warning(f"Could not encode frame to JPEG base64: {exc}")
        return ""


def extract_video_frames(
    video_path: Path, frame_stride: int = CFG.inference_frame_stride
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract, convert RGB, resize, and normalize frames from a video file.
    EXACT REPLICA of extract_video_frames in notebook.

    Args:
        video_path: Path to the input video file.
        frame_stride: Stride interval for frame sampling (default = 10).

    Returns:
        Tuple of (list of float32 normalized frames [H, W, C], video metadata dictionary).

    Raises:
        CorruptedVideoError: If OpenCV fails to open or read video frames.
        ValueError: If video contains fewer frames than required sequence length.
    """
    if not video_path.exists():
        raise CorruptedVideoError(f"Video file not found at: {video_path}")

    metadata = get_video_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CorruptedVideoError(f"Could not open video file: {video_path}")

    frames: List[np.ndarray] = []
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_AREA
                )
                frames.append(frame.astype(np.float32) / 255.0)
            frame_id += 1
    finally:
        cap.release()

    if len(frames) < CFG.sequence_length:
        raise ValueError(
            f"Video has only {len(frames)} sampled frames; need at least {CFG.sequence_length} "
            f"(sampled every {frame_stride} frames)."
        )

    return frames, metadata


def frames_to_sequences(
    frames: List[np.ndarray],
    sequence_length: int = CFG.sequence_length,
    stride: int = CFG.inference_sequence_stride,
) -> np.ndarray:
    """
    Convert list of frame arrays into sliding window 5D sequences tensor.
    EXACT REPLICA of frames_to_sequences in notebook.

    Args:
        frames: List of normalized frame arrays [H, W, C].
        sequence_length: Number of frames per temporal sequence (default = 16).
        stride: Step size between consecutive sequences (default = 8).

    Returns:
        Numpy array of shape [N, sequence_length, image_size, image_size, channels].
    """
    sequences = []
    for start in range(0, len(frames) - sequence_length + 1, stride):
        sequences.append(np.stack(frames[start : start + sequence_length], axis=0))
    
    if not sequences:
        sequences.append(np.stack(frames[:sequence_length], axis=0))
        
    return np.stack(sequences, axis=0).astype(np.float32)


def generate_anomaly_timeline(
    probabilities: np.ndarray,
    frames: List[np.ndarray],
    metadata: Dict[str, Any],
    sequence_length: int = CFG.sequence_length,
    seq_stride: int = CFG.inference_sequence_stride,
    frame_stride: int = CFG.inference_frame_stride,
) -> List[Dict[str, Any]]:
    """
    Build temporal timeline mapping sequence window probabilities to video timestamps (seconds)
    and representative frame base64 thumbnails.

    Args:
        probabilities: Model output array for sequences [N, num_classes].
        frames: Sampled video frames list.
        metadata: Video metadata including FPS and duration.
        sequence_length: Frame count per sequence window.
        seq_stride: Window step in sampled frames.
        frame_stride: Video frame sampling rate.

    Returns:
        List of timeline segment dictionaries with start/end times, thumbnails, and anomaly labels.
    """
    fps = metadata.get("fps", 30.0)
    timeline: List[Dict[str, Any]] = []

    for i, seq_probs in enumerate(probabilities):
        start_sampled_frame = i * seq_stride
        end_sampled_frame = start_sampled_frame + sequence_length

        start_sec = round((start_sampled_frame * frame_stride) / fps, 2)
        end_sec = round((end_sampled_frame * frame_stride) / fps, 2)

        pred_id = int(np.argmax(seq_probs))
        pred_class = CFG.id_to_class[pred_id]
        confidence = float(seq_probs[pred_id])
        is_anomaly = pred_class != "NormalVideos"

        # Calculate combined anomaly score (sum of all non-normal class probabilities)
        normal_idx = CFG.class_to_id.get("NormalVideos", 7)
        anomaly_score = float(1.0 - seq_probs[normal_idx])

        # Representative thumbnail frame from the middle of the window
        mid_frame_idx = min(start_sampled_frame + sequence_length // 2, len(frames) - 1)
        thumbnail_b64 = frame_to_base64_jpeg(frames[mid_frame_idx])

        timeline.append({
            "segment_id": i + 1,
            "start_time": start_sec,
            "end_time": end_sec,
            "time_range": f"{start_sec:.1f} - {end_sec:.1f} sec",
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "thumbnail": thumbnail_b64,
        })

    return timeline


def predict_video(video_path: Path) -> Dict[str, Any]:
    """
    Complete Video Anomaly Detection Inference Pipeline.

    Steps:
    1. Extract and preprocess frames.
    2. Construct sequence sliding windows.
    3. Run batch prediction using singleton ModelLoader.
    4. Aggregate mean probability across sequences.
    5. Derive predicted class, confidence, top-3 classes, timeline, and frame thumbnails.

    Args:
        video_path: Path to video file.

    Returns:
        JSON-serializable prediction result dictionary.
    """
    logger.info(f"Starting inference pipeline on video: {video_path}")
    
    # 1. Preprocess & Extract Frames
    frames, metadata = extract_video_frames(video_path, frame_stride=CFG.inference_frame_stride)
    
    # 2. Form Sequences
    sequences = frames_to_sequences(
        frames, sequence_length=CFG.sequence_length, stride=CFG.inference_sequence_stride
    )
    
    # 3. Model Inference (Singleton instance)
    model_loader = ModelLoader.get_instance()
    probabilities = model_loader.predict(sequences, batch_size=CFG.batch_size)
    
    # 4. Aggregate Probabilities
    mean_probability = probabilities.mean(axis=0)
    pred_id = int(np.argmax(mean_probability))
    predicted_class = CFG.id_to_class[pred_id]
    confidence = float(mean_probability[pred_id])
    
    # 5. Calculate Top-3 Predictions
    top3_indices = np.argsort(mean_probability)[::-1][:3]
    top3_predictions = [
        {
            "class": CFG.id_to_class[idx],
            "confidence": round(float(mean_probability[idx]), 4),
            "percentage": f"{float(mean_probability[idx]) * 100:.1f}%",
        }
        for idx in top3_indices
    ]
    
    # 6. Generate Anomaly Timeline with Base64 Thumbnails
    timeline = generate_anomaly_timeline(
        probabilities, frames, metadata, sequence_length=CFG.sequence_length
    )
    
    # 7. Formulate All Class Probabilities
    all_probabilities = {
        CFG.id_to_class[i]: round(float(mean_probability[i]), 4)
        for i in range(CFG.num_classes)
    }

    result = {
        "status": "success",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "is_anomaly": predicted_class != "NormalVideos",
        "top3": top3_predictions,
        "probabilities": all_probabilities,
        "timeline": timeline,
        "video_metadata": {
            "num_sampled_frames": len(frames),
            "num_sequences": len(sequences),
            "duration_sec": metadata["duration_sec"],
            "resolution": f"{metadata['width']}x{metadata['height']}",
            "fps": metadata["fps"],
        },
    }
    
    logger.info(
        f"Inference complete. Prediction: {predicted_class} ({confidence * 100:.2f}%) "
        f"for {len(sequences)} sequence windows."
    )
    return result
