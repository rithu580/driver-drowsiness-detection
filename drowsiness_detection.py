import cv2
import mediapipe as mp
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db
import os

# ================= FIREBASE INIT =================
firebase_enabled = False
if os.path.exists("firebase_key.json"):
    try:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://drivertrackingproject-default-rtdb.firebaseio.com/"
        })
        ref = db.reference("drivers/driver_1")
        firebase_enabled = True
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
else:
    print("Firebase key file not found. Running without Firebase.")

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# ================= LANDMARK INDEXES =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]        # upper & lower lip
NOSE = 1
CHIN = 152

# ================= FUNCTIONS =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_open_ratio(top, bottom):
    return abs(top[1] - bottom[1])

def head_tilt_angle(nose, chin):
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    return np.degrees(np.arctan2(dx, dy))

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

eye_closed_time = 0
yawn_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ACTIVE"
    alert_msg = "Normal"

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ---- Eye ----
        left_eye = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in LEFT_EYE])
        right_eye = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in RIGHT_EYE])
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        # ---- Mouth ----
        mouth_top = (int(lm[MOUTH[0]].x*w), int(lm[MOUTH[0]].y*h))
        mouth_bottom = (int(lm[MOUTH[1]].x*w), int(lm[MOUTH[1]].y*h))
        mouth_ratio = mouth_open_ratio(mouth_top, mouth_bottom)

        # ---- Head tilt ----
        nose = (int(lm[NOSE].x*w), int(lm[NOSE].y*h))
        chin = (int(lm[CHIN].x*w), int(lm[CHIN].y*h))
        tilt = head_tilt_angle(nose, chin)

        # ================= LOGIC =================
        if ear < 0.25:
            eye_closed_time += 0.1
        else:
            eye_closed_time = 0

        if mouth_ratio > 25:
            yawn_time += 0.1
        else:
            yawn_time = 0

        if eye_closed_time > 2:
            status = "DROWSY"
            alert_msg = "Eyes closed too long"

        if yawn_time > 1.5:
            status = "DROWSY"
            alert_msg = "Yawning detected"

        if abs(tilt) > 20:
            status = "DROWSY"
            alert_msg = "Head tilt detected"

        # ================= FIREBASE UPDATE =================
        if firebase_enabled:
            ref.update({
                "status": status,
                "alert": alert_msg,
                "timestamp": time.time()
            })

        # ================= DISPLAY =================
        cv2.putText(frame, f"STATUS: {status}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.putText(frame, alert_msg, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
