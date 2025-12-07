import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
import time

# ---------- optional text to speech (webcam only) ----------
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 180)
except ImportError:
    tts_engine = None


def speak(text, enabled: bool):
    if not enabled or tts_engine is None:
        return
    if text:
        tts_engine.say(text)
        tts_engine.runAndWait()


# ====================================================
# GEOMETRY
# ====================================================
def calculate_angle(a, b, c):
    """Angle (deg) at point b formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


# ====================================================
# FORM RULES
# ====================================================
def rule_elbow_bicep(angle):
    """Elbow form rule for bicep curls."""
    if angle > 150:
        return "Arm straight – start curl", False
    elif 100 < angle <= 150:
        return "Half curl – lift more", False
    elif 40 <= angle <= 100:
        return "Good curl range", True
    elif angle < 40:
        return "Full curl – excellent!", True
    else:
        return "Unclear elbow posture", False


def rule_wrist_shoulder_lateral(wrist, shoulder):
    """Wrist–shoulder alignment rule."""
    dy = abs(wrist[1] - shoulder[1])
    if dy < 40:
        return "Wrist at shoulder height – good raise", True
    elif dy < 80:
        return "Lift arm a bit higher", False
    return "Arm too low at top", False


def rule_back_posture(ls, rs, lh, rh):
    """Loose back posture rule."""
    shoulder_tilt = abs(ls[1] - rs[1])
    hip_tilt = abs(lh[1] - rh[1])

    if shoulder_tilt <= 60 and hip_tilt <= 60:
        return "Back straight & symmetric (OK)", True
    if shoulder_tilt <= 120 or hip_tilt <= 120:
        return "Slight lean — acceptable", True
    return "Shoulders uneven — avoid leaning", False


# ====================================================
# LATERAL RAISE ARM TRACKER (one arm)
# ====================================================
def update_lateral_arm(
    sh_angle,
    wrist,
    shoulder,
    elbow_angle,
    back_ok,
    stage,
    ready,
    reps,
    good,
    bad,
    top_good,
):

    DOWN_T = 35.0
    UP_T = 75.0

    elbow_ok = elbow_angle > 150
    wrist_msg, wrist_ok = rule_wrist_shoulder_lateral(wrist, shoulder)
    top_position = 70 <= sh_angle <= 100

    if stage is None:
        stage = "down" if sh_angle < DOWN_T else "up"
        ready = sh_angle < DOWN_T
        top_good = False

    if sh_angle < DOWN_T:
        ready = True

    if ready and sh_angle > UP_T:
        if stage == "down":
            stage = "up"
            top_good = wrist_ok and elbow_ok and back_ok and top_position
        else:
            top_good = top_good or (wrist_ok and elbow_ok and back_ok and top_position)

    if stage == "up" and sh_angle < DOWN_T:
        stage = "down"
        ready = False
        reps += 1
        if top_good:
            good += 1
        else:
            bad += 1
        top_good = False

    return (
        stage,
        ready,
        reps,
        good,
        bad,
        top_good,
        wrist_msg,
    )


# ====================================================
# FRAME PROCESSOR
# ====================================================
def process_frame(
    exercise,
    frame,
    pose,
    mp_pose,
    mp_draw,
    elbow_smooth_R,
    prev_angle_R,
    prev_sign_R,
    top_angle_R,
    bottom_angle_R,
    bottom_good_R,
    curl_reps,
    curl_good,
    curl_bad,
    L_stage,
    L_ready,
    L_reps,
    L_good,
    L_bad,
    L_top_good,
    R_stage,
    R_ready,
    R_reps,
    R_good,
    R_bad,
    R_top_good,
):

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    overlay = []
    elbow_angle_R = None
    back_ok = True

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Keypoints
        L_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
        R_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])

        L_el = np.array([lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h])
        R_el = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                         lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h])

        L_wr = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                         lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
        R_wr = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                         lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])

        L_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                          lm[mp_pose.PoseLandmark.LEFT_HIP].y * h])
        R_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])

        # Back rule
        back_msg, back_ok = rule_back_posture(L_sh, R_sh, L_hip, R_hip)
        overlay.append(f"Back: {back_msg}")

        # ========================================================
        # BICEP CURL
        # ========================================================
        elbow_raw_R = calculate_angle(R_sh, R_el, R_wr)
        elbow_smooth_R.append(elbow_raw_R)
        elbow_angle_R = sum(elbow_smooth_R) / len(elbow_smooth_R)

        if "Curl" in exercise:

            elbow_msg, elbow_ok = rule_elbow_bicep(elbow_angle_R)
            overlay.append(f"Elbow: {elbow_msg}")
            overlay.append(f"Elbow Angle: {elbow_angle_R:.1f}°")

            if prev_angle_R is not None:
                diff = elbow_angle_R - prev_angle_R

                if abs(diff) < 2:
                    sign = 0
                else:
                    sign = -1 if diff < 0 else 1

                # lowering → curling (top)
                if prev_sign_R == 1 and sign == -1:
                    top_angle_R = prev_angle_R

                # curling → lowering (bottom)
                if prev_sign_R == -1 and sign == 1:
                    bottom_angle_R = prev_angle_R
                    _, bottom_ok = rule_elbow_bicep(bottom_angle_R)
                    bottom_good_R = bottom_ok and back_ok

                    if top_angle_R is not None:
                        amp = top_angle_R - bottom_angle_R
                        if (
                            amp >= 25
                            and bottom_angle_R <= 90
                            and top_angle_R >= 140
                        ):
                            curl_reps += 1
                            if bottom_good_R:
                                curl_good += 1
                            else:
                                curl_bad += 1
                        top_angle_R = None
                        bottom_angle_R = None

                if sign != 0:
                    prev_sign_R = sign

            prev_angle_R = elbow_angle_R

            overlay.append(f"Curl Reps: {curl_reps} (Good: {curl_good}, Wrong: {curl_bad})")

        # ========================================================
        # LATERAL RAISE
        # ========================================================
        if "Lateral" in exercise:

            v_L = np.array([L_sh[0], L_sh[1] + 150])
            v_R = np.array([R_sh[0], R_sh[1] + 150])

            sh_angle_L = calculate_angle(v_L, L_sh, L_el)
            sh_angle_R = calculate_angle(v_R, R_sh, R_el)

            el_angle_L = calculate_angle(L_sh, L_el, L_wr)
            el_angle_R = calculate_angle(R_sh, R_el, R_wr)

            # LEFT ARM update
            (
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
                L_msg,
            ) = update_lateral_arm(
                sh_angle_L,
                L_wr,
                L_sh,
                el_angle_L,
                back_ok,
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
            )

            # RIGHT ARM update
            (
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
                R_msg,
            ) = update_lateral_arm(
                sh_angle_R,
                R_wr,
                R_sh,
                el_angle_R,
                back_ok,
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
            )

            overlay.append(f"L-Shoulder: {sh_angle_L:.1f}° | {L_msg}")
            overlay.append(f"R-Shoulder: {sh_angle_R:.1f}° | {R_msg}")
            overlay.append(f"L Reps: {L_reps} (Good: {L_good}, Wrong: {L_bad})")
            overlay.append(f"R Reps: {R_reps} (Good: {R_good}, Wrong: {R_bad})")

    # -------- Overlay drawing --------
    y = 30
    for t in overlay:
        cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30

    return (
        frame,
        elbow_angle_R,
        back_ok,
        prev_angle_R,
        prev_sign_R,
        top_angle_R,
        bottom_angle_R,
        bottom_good_R,
        curl_reps,
        curl_good,
        curl_bad,
        L_stage,
        L_ready,
        L_reps,
        L_good,
        L_bad,
        L_top_good,
        R_stage,
        R_ready,
        R_reps,
        R_good,
        R_bad,
        R_top_good,
    )


# ====================================================
# STREAMLIT UI
# ====================================================
st.title("💪 Exercise Form Analyzer")

exercise = st.selectbox(
    "Select Exercise",
    ["Right Bicep Curl", "Lateral Raise (Both Arms)"],
    key="exercise_selector"
)

mode = st.radio(
    "Select Input Mode",
    ["Upload Video", "Webcam Live"],
    key="mode_selector"
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ====================================================
# UPLOAD VIDEO MODE
# ====================================================
if mode == "Upload Video":

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if uploaded:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded.read())

        st.video(video_path)

        frame_box = st.empty()
        chart_box = st.empty()
        summary_box = st.empty()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps else 0.03

        # Save output video
        w = int(cap.get(3))
        h = int(cap.get(4))
        out = cv2.VideoWriter("demo_output.mp4",
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps,
                              (w, h))

        elbow_smooth_R = deque(maxlen=7)
        elbow_ts = []
        back_ts = []

        # Curl state
        prev_angle_R = None
        prev_sign_R = 0
        top_angle_R = None
        bottom_angle_R = None
        bottom_good_R = False
        curl_reps = curl_good = curl_bad = 0

        # Lateral state
        L_stage = None
        L_ready = False
        L_reps = L_good = L_bad = 0
        L_top_good = False
        R_stage = None
        R_ready = False
        R_reps = R_good = R_bad = 0
        R_top_good = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            (
                frame,
                elbow_angle_R,
                back_ok,
                prev_angle_R,
                prev_sign_R,
                top_angle_R,
                bottom_angle_R,
                bottom_good_R,
                curl_reps,
                curl_good,
                curl_bad,
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
            ) = process_frame(
                exercise,
                frame,
                pose,
                mp_pose,
                mp_draw,
                elbow_smooth_R,
                prev_angle_R,
                prev_sign_R,
                top_angle_R,
                bottom_angle_R,
                bottom_good_R,
                curl_reps,
                curl_good,
                curl_bad,
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
            )

            out.write(frame)

            if elbow_angle_R is not None:
                elbow_ts.append(elbow_angle_R)
            back_ts.append(1 if back_ok else 0)

            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(delay)

        cap.release()
        out.release()

        back_pct = 100 * sum(back_ts) / max(1, len(back_ts))

        if "Curl" in exercise:
            summary_box.markdown(
                f"""
                ## Bicep Curl Summary  
                - Reps: **{curl_reps}**  
                - Good: **{curl_good}**  
                - Wrong: **{curl_bad}**  
                - Back posture OK: **{back_pct:.1f}%**
                """
            )
        else:
            total_reps = L_reps + R_reps
            total_good = L_good + R_good
            total_bad = L_bad + R_bad
            summary_box.markdown(
                f"""
                ## Lateral Raise Summary  
                - Left: {L_reps} (Good: {L_good}, Wrong: {L_bad})  
                - Right: {R_reps} (Good: {R_good}, Wrong: {R_bad})  
                - Total: {total_reps} (Good: {total_good}, Wrong: {total_bad})  
                - Back posture OK: **{back_pct:.1f}%**
                """
            )

        # Angle chart
        if elbow_ts:
            st.subheader("Elbow Angle Over Time")
            fig, ax = plt.subplots()
            ax.plot(elbow_ts)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Angle (deg)")
            chart_box.pyplot(fig)


# ====================================================
# WEBCAM MODE
# ====================================================
elif mode == "Webcam Live":

    st.write("Press Start Webcam to begin.")

    if st.button("Start Webcam"):

        frame_box = st.empty()
        info_box = st.empty()

        cap = cv2.VideoCapture(0)

        elbow_smooth_R = deque(maxlen=7)

        prev_angle_R = None
        prev_sign_R = 0
        top_angle_R = None
        bottom_angle_R = None
        bottom_good_R = False
        curl_reps = curl_good = curl_bad = 0

        L_stage = None
        L_ready = False
        L_reps = L_good = L_bad = 0
        L_top_good = False

        R_stage = None
        R_ready = False
        R_reps = R_good = R_bad = 0
        R_top_good = False

        last_L = last_R = last_C = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            (
                frame,
                elbow_angle_R,
                back_ok,
                prev_angle_R,
                prev_sign_R,
                top_angle_R,
                bottom_angle_R,
                bottom_good_R,
                curl_reps,
                curl_good,
                curl_bad,
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
            ) = process_frame(
                exercise,
                frame,
                pose,
                mp_pose,
                mp_draw,
                elbow_smooth_R,
                prev_angle_R,
                prev_sign_R,
                top_angle_R,
                bottom_angle_R,
                bottom_good_R,
                curl_reps,
                curl_good,
                curl_bad,
                L_stage,
                L_ready,
                L_reps,
                L_good,
                L_bad,
                L_top_good,
                R_stage,
                R_ready,
                R_reps,
                R_good,
                R_bad,
                R_top_good,
            )

            # -------- Voice feedback --------
            if "Curl" in exercise:
                if curl_reps > last_C:
                    speak("Good curl" if curl_good >= curl_bad else "Fix curl form", True)
                    last_C = curl_reps
            else:
                if L_reps > last_L:
                    speak("Good left raise" if L_good >= L_bad else "Fix left raise", True)
                    last_L = L_reps
                if R_reps > last_R:
                    speak("Good right raise" if R_good >= R_bad else "Fix right raise", True)
                    last_R = R_reps

            # -------- Display --------
            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if "Curl" in exercise:
                info_box.markdown(f"### Curl Reps: {curl_reps} (Good: {curl_good}, Wrong: {curl_bad})")
            else:
                info_box.markdown(
                    f"""
                    ### Lateral Raise  
                    Left: {L_reps} (Good: {L_good}, Wrong: {L_bad})  
                    Right: {R_reps} (Good: {R_good}, Wrong: {R_bad})
                    """
                )

            time.sleep(0.03)

        cap.release()
