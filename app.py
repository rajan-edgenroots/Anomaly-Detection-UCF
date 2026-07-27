"""
Production Flask Web Application for Federated Video Anomaly Detection.
Runs directly on Hugging Face Spaces via `python app.py`.
Exposes interactive modern UI, video streaming endpoint `/uploads/<filename>`, and REST API `POST /predict`.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from config import CFG
from model import ModelLoader
from predict import predict_video
from utils import (
    validate_file_extension,
    validate_video_url,
    download_video_from_url,
    prune_old_uploads,
    VideoProcessingError,
    InvalidURLError,
    InvalidFileError,
    CorruptedVideoError,
    DownloadError,
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = CFG.max_content_length
app.config["UPLOAD_FOLDER"] = CFG.upload_folder
app.config["TEMP_FOLDER"] = CFG.temp_folder

# Create required directories
CFG.upload_folder.mkdir(parents=True, exist_ok=True)
CFG.temp_folder.mkdir(parents=True, exist_ok=True)

# Pre-load singleton model ONCE at server startup
logger.info("Pre-loading Federated Video Anomaly Detection model at startup...")
try:
    _ = ModelLoader.get_instance()
    logger.info("Model pre-loaded and ready for predictions.")
except Exception as exc:
    logger.critical(f"Failed to load model singleton at startup: {exc}")


@app.route("/", methods=["GET"])
def index():
    """Render single-page web UI."""
    return render_template("index.html")


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename: str):
    """
    Serve uploaded/downloaded video files directly for browser playback.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API Endpoint & Frontend Handler: POST /predict

    Accepts:
    1. Multipart file upload: `video` or `file` field (.mp4, .avi, .mov, .mkv up to 500MB).
    2. JSON payload or Form data: `video_url`, `url`, or `kaggle_url` (direct video link or Cloud Storage URL).

    Returns:
        JSON response with prediction, confidence, top-3 classes, timeline with frame thumbnails, and video_url.
    """
    target_video_path: Path = None

    try:
        # Periodic pruning of old upload files (>1 hour)
        prune_old_uploads(CFG.upload_folder, max_age_seconds=3600)

        # Case A: Video URL provided
        video_url = None
        if request.is_json and request.json:
            video_url = request.json.get("video_url") or request.json.get("url") or request.json.get("kaggle_url")
        elif request.form:
            video_url = request.form.get("video_url") or request.form.get("url") or request.form.get("kaggle_url")

        if video_url:
            logger.info(f"Processing Video URL input: {video_url[:120]}...")
            validated_url = validate_video_url(video_url)
            # Save downloaded video in upload_folder so Flask can serve it to the HTML5 video player
            target_video_path = download_video_from_url(validated_url, CFG.upload_folder)

        # Case B: Video file upload provided
        elif "video" in request.files or "file" in request.files:
            file = request.files.get("video") or request.files.get("file")
            if not file or file.filename == "":
                return jsonify({"status": "error", "message": "No file selected."}), 400

            ext = validate_file_extension(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            target_video_path = CFG.upload_folder / unique_filename
            file.save(str(target_video_path))

        else:
            return jsonify({
                "status": "error",
                "message": "Please provide either a video file upload or a valid Video URL."
            }), 400

        # Execute Prediction Pipeline
        results = predict_video(target_video_path)

        # Generate web accessible URL for video player playback
        web_video_url = f"/uploads/{target_video_path.name}"

        # Standardize Output Format
        response_payload = {
            "status": "success",
            "prediction": results["predicted_class"],
            "confidence": results["confidence"],
            "confidence_percentage": results["confidence_percentage"],
            "is_anomaly": results["is_anomaly"],
            "top3": results["top3"],
            "probabilities": results["probabilities"],
            "timeline": results["timeline"],
            "video_metadata": results["video_metadata"],
            "video_url": web_video_url,
        }
        return jsonify(response_payload), 200

    except InvalidURLError as exc:
        logger.warning(f"URL Validation Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 400

    except InvalidFileError as exc:
        logger.warning(f"File Validation Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 415

    except CorruptedVideoError as exc:
        logger.error(f"Corrupted Video Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 400

    except DownloadError as exc:
        logger.error(f"Download Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 400

    except Exception as exc:
        logger.exception(f"Unexpected Error during prediction: {exc}")
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred during prediction: {str(exc)}"
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle oversized file upload error (max 500 MB)."""
    return jsonify({
        "status": "error",
        "message": "File exceeds maximum upload size limit of 500 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 routes."""
    return jsonify({"status": "error", "message": "Resource not found."}), 404


if __name__ == "__main__":
    # Hugging Face Spaces default port is 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
