from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

# OpenCV VideoCapture object to capture the camera feed
cap = cv2.VideoCapture('rtsp://0.tcp.ap.ngrok.io:11792/user:1cinnovation;pwd:1cinnovation123')

def video_stream():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                socketio.emit('video_frame', frame_bytes)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(video_stream)

if __name__ == '__main__':
    socketio.run(app, debug=True)
