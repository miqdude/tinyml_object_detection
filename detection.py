import cv2
import numpy as np
from flask import Flask, Response
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = "yolov5nano320.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
H, W = input_shape[1], input_shape[2]

# --------------------------
# Initialize camera
# --------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera.")


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # --------------------------
        # Preprocess
        # --------------------------
        img = cv2.resize(frame, (W, H))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        if input_details[0]["dtype"] == np.uint8:
            img = img.astype(np.uint8)

        # --------------------------
        # Inference
        # --------------------------
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # --------------------------
        # Draw prediction on frame
        # --------------------------
        text = f"Pred: {output.flatten()[0]:.3f}" \
            if output.ndim == 2 or output.ndim == 1 else "OK"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --------------------------
        # Encode frame as JPEG
        # --------------------------
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream as multipart jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return "<h2>TFLite Camera Stream</h2><img src='/stream'>"


@app.route('/stream')
def stream():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
