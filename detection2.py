from picamera2 import Picamera2
import cv2
import numpy as np
from flask import Flask, Response
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = "model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]
H, W = input_shape[1], input_shape[2]

# --------------------------
# Initialize Raspberry Pi Camera
# --------------------------
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()


def gen_frames():
    while True:
        # Capture frame (as RGB888)
        frame = picam2.capture_array()

        # --------------------------
        # Preprocess for TFLite
        # --------------------------
        img = cv2.resize(frame, (W, H)).astype(np.float32)
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
        # Draw prediction
        # --------------------------
        text = f"Pred: {output.flatten()[0]:.3f}" if output.ndim >= 1 else "OK"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --------------------------
        # Stream frame as JPEG
        # --------------------------
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

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
