---
title: Federated Video Anomaly Detection
emoji: 🎥
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Federated Video Anomaly Detection (SE-ResNet18)

Production-ready Hugging Face Space for Video Anomaly Detection powered by a Federated Learning trained SE-ResNet18 sequence model.

## Features
- **Inference Only**: Zero training or dataset code; loads pre-trained model weights once on startup.
- **Allowed Inputs**: Supports direct file upload (`.mp4`, `.avi`, `.mov`, `.mkv` up to 500 MB) OR Kaggle Dataset URLs (`https://www.kaggle.com/datasets/...`). Rejects external non-Kaggle domains.
- **Preprocessing Integrity**: Frame sampling (stride 10), resize (64x64), RGB conversion, normalization, and sequence length (16 frames, stride 8) strictly match the notebook training setup.
- **Outputs**: Overall predicted class, main confidence score, Top-3 prediction probabilities, second-by-second temporal anomaly timeline, and video player seek sync.
- **REST API**: Serves `POST /predict` returning structured JSON predictions.

## Local & Hugging Face Execution

Run directly via:
```bash
python app.py
```
App starts on port `7860`.

## REST API Specification

### `POST /predict`

#### File Upload Example:
```bash
curl -X POST -F "file=@/path/to/video.mp4" http://localhost:7860/predict
```

#### Kaggle URL Example:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"kaggle_url": "https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance"}' \
     http://localhost:7860/predict
```

#### Sample Response:
```json
{
  "status": "success",
  "prediction": "Robbery",
  "confidence": 0.934,
  "confidence_percentage": "93.4%",
  "is_anomaly": true,
  "top3": [
    { "class": "Robbery", "confidence": 0.934, "percentage": "93.4%" },
    { "class": "Stealing", "confidence": 0.045, "percentage": "4.5%" },
    { "class": "NormalVideos", "confidence": 0.012, "percentage": "1.2%" }
  ],
  "timeline": [
    { "segment_id": 1, "start_time": 0.0, "end_time": 5.3, "predicted_class": "NormalVideos", "is_anomaly": false, "confidence": 0.89 },
    { "segment_id": 2, "start_time": 5.3, "end_time": 10.6, "predicted_class": "Robbery", "is_anomaly": true, "confidence": 0.94 }
  ],
  "video_metadata": {
    "num_sampled_frames": 120,
    "num_sequences": 14,
    "duration_sec": 40.0,
    "resolution": "1280x720",
    "fps": 30.0
  }
}
```
# Anomaly-Detection-UCF
