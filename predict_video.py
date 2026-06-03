import cv2, mediapipe as mp, numpy as np, joblib
from collections import deque

# ------------------ Load model ------------------
clf = joblib.load("exercise_classifier.pkl")

# ------------------ MediaPipe ------------------
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ------------------ Video ------------------
VIDEO = r"Z:\Main\2ndtime\data\squat.mp4"
cap = cv2.VideoCapture(VIDEO)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ------------------ Landmark indices ------------------
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
NOSE = 0

# ------------------ Helpers ------------------
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# ------------------ Feature extraction ------------------
def extract_features(seq):
    seq = np.array(seq).reshape(-1,33,3)
    arm = ((seq[:,LW,1]+seq[:,RW,1])/2)-((seq[:,LS,1]+seq[:,RS,1])/2)
    legs = np.linalg.norm(seq[:,LA,:2]-seq[:,RA,:2],axis=1)
    knee = ((seq[:,LK,1]+seq[:,RK,1])/2)-((seq[:,LH,1]+seq[:,RH,1])/2)
    vert = seq[:,NOSE,1]
    return np.array([
        arm.mean(), arm.std(), arm.max(),
        legs.mean(), legs.std(),
        knee.mean(), knee.std(), knee.max(),
        vert.std()
    ]).reshape(1,-1)

# ------------------ UI State ------------------
mode = "AUTO"
manual_ex = "Jumping Jacks"
rep_count = 0
state = "up"

window = deque(maxlen=25)
pred_hist = deque(maxlen=10)

label_map = {0:"Other",1:"Jumping Jacks",2:"Squats",3:"High Knees"}

# ------------------ UI Buttons ------------------
btn_y1 = int(H*0.83)
btn_y2 = int(H*0.97)

buttons = {
    "AUTO": (int(W*0.02), btn_y1, int(W*0.15), btn_y2),
    "JJ":   (int(W*0.18), btn_y1, int(W*0.31), btn_y2),
    "SQ":   (int(W*0.34), btn_y1, int(W*0.47), btn_y2),
    "HK":   (int(W*0.50), btn_y1, int(W*0.63), btn_y2)
}

def click(event,x,y,flags,param):
    global mode, manual_ex, rep_count
    if event==cv2.EVENT_LBUTTONDOWN:
        for k,(x1,y1,x2,y2) in buttons.items():
            if x1<x<x2 and y1<y<y2:
                if k=="AUTO": mode="AUTO"
                if k=="JJ": manual_ex="Jumping Jacks"; mode="MANUAL"; rep_count=0
                if k=="SQ": manual_ex="Squats"; mode="MANUAL"; rep_count=0
                if k=="HK": manual_ex="High Knees"; mode="MANUAL"; rep_count=0

cv2.namedWindow("Rehab MVP")
cv2.setMouseCallback("Rehab MVP", click)

# ------------------ Main Loop ------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)

    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp_style.get_default_pose_landmarks_style()
        )

    exercise = "None"
    conf = 0

    if res.pose_landmarks:
        pts=[]
        for lm in res.pose_landmarks.landmark:
            pts.extend([lm.x,lm.y,lm.z])
        window.append(pts)

        if len(window)==25:
            feats = extract_features(window)
            probs = clf.predict_proba(feats)[0]
            conf = np.max(probs)
            pred = np.argmax(probs)
            pred_hist.append(pred)
            smooth = max(set(pred_hist), key=pred_hist.count)

            exercise = label_map[smooth] if mode=="AUTO" else manual_ex

            seq = np.array(window).reshape(-1,33,3)

            # -------- Rep Logic --------
            if exercise=="Jumping Jacks":
                l = angle(seq[-1,LE],seq[-1,LS],seq[-1,LH])
                r = angle(seq[-1,RE],seq[-1,RS],seq[-1,RH])
                a=(l+r)/2
                if a<60 and state=="up": state="down"
                if a>140 and state=="down": rep_count+=1; state="up"

            if exercise == "Squats":
                # vertical hip motion
                hip_y = (seq[-1, LH, 1] + seq[-1, RH, 1]) / 2

                # knee bend
                knee_angle = (
                    angle(seq[-1, LH], seq[-1, LK], seq[-1, LA]) +
                    angle(seq[-1, RH], seq[-1, RK], seq[-1, RA])
                ) / 2

                # adaptive thresholds
                if hip_y > 0.55 and knee_angle < 140:
                    state = "down"

                if hip_y < 0.48 and state == "down":
                    rep_count += 1
                    state = "up"


            if exercise == "High Knees":
                left_knee = seq[-1, LK, 1]
                right_knee = seq[-1, RK, 1]
                hip = (seq[-1, LH, 1] + seq[-1, RH, 1]) / 2

                if left_knee < hip - 0.05 and state != "left":
                    rep_count += 1
                    state = "left"

                elif right_knee < hip - 0.05 and state != "right":
                    rep_count += 1
                    state = "right"

                if left_knee > hip and right_knee > hip:
                    state = "reset"


    # ------------------ UI ------------------
    cv2.rectangle(frame,(0,0),(W,int(H*0.18)),(0,0,0),-1)
    cv2.putText(frame,f"Mode: {mode}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(frame,f"Exercise: {exercise}",(200,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(frame,f"Reps: {rep_count}",(W-150,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    for k,(x1,y1,x2,y2) in buttons.items():
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,k,(x1+10,y1+30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("Rehab MVP",frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()
