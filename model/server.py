from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import logging

from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

app = Flask(__name__)
CORS(app)

# Initialize Socket.IO (websocket) support.
# Make sure you have: pip install flask-socketio eventlet
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)

# Load Models
face_detector = YOLOv5("saved_models/yolov5s-face.onnx")
anti_spoof = AntiSpoof("saved_models/AntiSpoofing_bin_1.5_128.onnx")


def decode_image(image_base64: str):
    """Convert Base64 string to OpenCV image (BGR)."""
    try:
        # In case it's a data URL: "data:image/png;base64,...."
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        logging.error(f"decode_image error: {e}")
        return None


def increased_crop(img, bbox, bbox_inc: float = 1.5):
    """
    Expand and crop bounding box area.
    bbox is assumed to be [x1, y1, x2, y2].
    """
    real_h, real_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    l = max(w, h)
    xc = x1 + w / 2
    yc = y1 + h / 2

    x = int(xc - l * bbox_inc / 2)
    y = int(yc - l * bbox_inc / 2)

    x1_new = max(0, x)
    y1_new = max(0, y)
    x2_new = min(real_w, x + int(l * bbox_inc))
    y2_new = min(real_h, y + int(l * bbox_inc))

    cropped = img[y1_new:y2_new, x1_new:x2_new]
    return cropped


def make_prediction(img):
    """
    Run YOLO + Anti-Spoof model.
    Always returns a 4-tuple: (bbox_or_None, label, score, cropped_face_or_None)
    """
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run face detector
        raw = face_detector([img_rgb])

        # Normalize output from YOLOv5 wrapper
        # It might be: list [detections_for_first_image, ...]
        # or np.ndarray directly. We handle both.
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return None, "FOCUS", 0.0, None
            bboxes = np.array(raw[0], dtype=float)
        else:
            bboxes = np.array(raw, dtype=float)

        # Handle empty or invalid shapes
        if bboxes is None or bboxes.size == 0:
            return None, "FOCUS", 0.0, None

        # If it has shape (N, M), take first row; if (M,), use directly
        if bboxes.ndim > 1:
            first = bboxes[0]
        else:
            first = bboxes

        first = np.array(first).flatten()
        if first.size < 4:
            # Not enough values to form a bbox
            return None, "FOCUS", 0.0, None

        # Use first 4 values as [x1, y1, x2, y2]
        bbox = first[:4].astype(int)
        cropped_img = increased_crop(img_rgb, bbox)

        # Anti-spoof prediction
        pred = anti_spoof([cropped_img])[0]
        score = float(pred[0][0])
        is_real = (np.argmax(pred) == 0 and score > 0.5)
        label = "REAL" if is_real else "FAKE"

        return bbox, label, score, (cropped_img if is_real else None)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None, "FOCUS", 0.0, None


def verify_face(face_img):
    """
    Dummy face verification.
    Replace this logic with DeepFace or a proper matcher if needed.
    """
    known = cv2.imread("image.png")
    if known is None or face_img is None:
        return False
    # TODO: real matching (e.g., DeepFace) can go here.
    return True


@app.route("/predict", methods=["POST"])
def predict():
    """HTTP Anti-Spoofing detection API."""
    try:
        data = request.get_json()
        image_base64 = data.get("image") if data else None

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        frame = decode_image(image_base64)
        if frame is None:
            return jsonify({"error": "Invalid Base64 image"}), 400

        bbox, label, score, cropped_face = make_prediction(frame)

        if bbox is None:
            return jsonify({"label": "FOCUS", "score": 0.0, "bbox": None})

        x1, y1, x2, y2 = map(int, bbox)

        return jsonify(
            {
                "label": label,
                "score": float(score),
                "bbox": [x1, y1, x2, y2],
            }
        )

    except Exception as e:
        logging.error(f"/predict Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/match", methods=["POST"])
def match():
    """HTTP Face matching API."""
    try:
        data = request.get_json()
        image_base64 = data.get("image") if data else None

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        frame = decode_image(image_base64)
        if frame is None:
            return jsonify({"error": "Invalid Base64 image"}), 400

        bbox, label, score, cropped_face = make_prediction(frame)

        # If no face, fake, or cropped_face missing → no match
        if bbox is None or label == "FAKE" or cropped_face is None:
            return jsonify({"match": False})

        match_result = verify_face(cropped_face)
        return jsonify({"match": bool(match_result)})

    except Exception as e:
        logging.error(f"/match Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


# --- Socket.IO handlers ---

@socketio.on("connect")
def _on_connect():
    logging.info(f"WebSocket client connected: {request.sid}")
    emit("connected", {"msg": "connected"})


@socketio.on("disconnect")
def _on_disconnect():
    logging.info(f"WebSocket client disconnected: {request.sid}")


@socketio.on("image")
def _on_image(data):
    """
    Handle incoming base64 image over websocket and emit a 'prediction' event.

    Expected incoming data: {'image': '<base64 or data URL>'}
    Emits: {'label': str, 'score': float, 'bbox': [x1,y1,x2,y2]} or {'error': '...'}
    """
    try:
        image_b64 = None
        if isinstance(data, dict):
            image_b64 = data.get("image")

        if not image_b64:
            emit("prediction", {"error": "no_image_provided"})
            return

        frame = decode_image(image_b64)
        if frame is None:
            emit("prediction", {"error": "invalid_image"})
            return

        bbox, label, score, _ = make_prediction(frame)

        if bbox is None:
            emit("prediction", {"label": "FOCUS", "score": 0.0, "bbox": []})
            return

        x1, y1, x2, y2 = map(int, bbox)
        emit(
            "prediction",
            {
                "label": label,
                "score": float(score),
                "bbox": [x1, y1, x2, y2],
            },
        )

    except Exception as e:
        logging.exception("Error processing websocket image")
        emit("prediction", {"error": "internal_error", "detail": str(e)})


if __name__ == "__main__":
    # Run with Socket.IO server so websockets work.
    # For production/dev with eventlet:
    #   pip install eventlet
    #   socketio.run(app, host="0.0.0.0", port=5000)
    socketio.run(app, host="0.0.0.0", port=5000)
