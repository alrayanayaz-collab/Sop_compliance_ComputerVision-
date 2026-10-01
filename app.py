import cv2
import sqlite3
import time
from datetime import datetime
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load the AI model. Replace 'yolov8n.pt' with 'best.pt' once you train your custom model.
model = YOLO('yolov8n.pt')

# Initialize the database to log violations
def init_db():
    conn = sqlite3.connect('sop_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, timestamp TEXT, issue TEXT, camera TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Global variables for temporal validation
consecutive_frames_with_violation = 0
VIOLATION_THRESHOLD = 30 # Number of frames the issue must persist before alerting

def generate_frames():
    global consecutive_frames_with_violation
    camera = cv2.VideoCapture(0) # 0 connects to your laptop's built-in webcam
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Run YOLO detection on the current frame
        results = model(frame, stream=True)
        violation_detected_in_frame = False
        
        for r in results:
            for box in r.boxes:
                # Get coordinates and class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Example Logic: If the model detects a 'no_head_cap' class
                # (For the default yolov8n model, class 0 is 'person'. We will use 'person' to test)
                if class_name == 'person': # Change 'person' to 'no_head_cap' for your final demo
                    violation_detected_in_frame = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Draw Red Box
                    cv2.putText(frame, "VIOLATION", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Temporal validation: Evaluate conditions against SOP rules over time to reduce false alerts 
        if violation_detected_in_frame:
            consecutive_frames_with_violation += 1
        else:
            consecutive_frames_with_violation = 0
            
        # Generate alert if violation persists beyond threshold
        if consecutive_frames_with_violation == VIOLATION_THRESHOLD:
            record_violation("Missing Head Cap", "Kitchen-Cam-01")
            
        # Encode the frame to send to the web browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def record_violation(issue, camera_id):
    conn = sqlite3.connect('sop_database.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO alerts (timestamp, issue, camera) VALUES (?, ?, ?)", (timestamp, issue, camera_id))
    conn.commit()
    conn.close()
    print(f"Alert Saved: {issue} at {timestamp}")

# Web Routes
@app.route('/')
def index():
    # Fetch violation history to display on the dashboard
    conn = sqlite3.connect('sop_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 10")
    recent_alerts = c.fetchall()
    conn.close()
    return render_template('index.html', alerts=recent_alerts)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True) 