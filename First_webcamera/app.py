import cv2
import time
import math
import mediapipe as mp

# =============================
# Webcam Posture Alert App
# Windows / Mac / Linux
# =============================
# q キーで終了

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

BAD_SECONDS_THRESHOLD = 3.0   # 悪い姿勢が何秒続いたら警告するか
FORWARD_HEAD_THRESHOLD = 0.08 # 肩中心から鼻がどれだけ横にズレたら悪い姿勢扱いにするか
SHOULDER_TILT_THRESHOLD = 0.06 # 左右肩の高さ差

bad_start_time = None
alert_active = False

def point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return lm.x, lm.y, lm.visibility

def main():
    global bad_start_time, alert_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした。別のアプリがカメラを使用していないか確認してください。")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラ映像を取得できませんでした。")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            status = "No person detected"
            score = 100
            is_bad = False
            reason = ""

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                nose_x, nose_y, nose_v = point(landmarks, "NOSE")
                ls_x, ls_y, ls_v = point(landmarks, "LEFT_SHOULDER")
                rs_x, rs_y, rs_v = point(landmarks, "RIGHT_SHOULDER")

                if min(nose_v, ls_v, rs_v) > 0.5:
                    shoulder_center_x = (ls_x + rs_x) / 2
                    shoulder_center_y = (ls_y + rs_y) / 2

                    # Webカメラ正面想定：鼻が肩中心から大きく左右にズレる、または肩の高さ差が大きいと悪化扱い
                    head_offset = abs(nose_x - shoulder_center_x)
                    shoulder_tilt = abs(ls_y - rs_y)

                    score -= int(head_offset * 500)
                    score -= int(shoulder_tilt * 500)
                    score = max(0, min(100, score))

                    if head_offset > FORWARD_HEAD_THRESHOLD:
                        is_bad = True
                        reason = "Head is not centered"
                    if shoulder_tilt > SHOULDER_TILT_THRESHOLD:
                        is_bad = True
                        reason = "Shoulders are tilted"

                    if is_bad:
                        if bad_start_time is None:
                            bad_start_time = time.time()
                        elapsed = time.time() - bad_start_time
                        if elapsed >= BAD_SECONDS_THRESHOLD:
                            alert_active = True
                            status = "POSTURE ALERT! Sit straight"
                        else:
                            status = f"Bad posture... {elapsed:.1f}s"
                    else:
                        bad_start_time = None
                        alert_active = False
                        status = "Good posture"
                        reason = ""

                    # 点と線を描画
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    # 肩中心と鼻を線で表示
                    nose_px = (int(nose_x * w), int(nose_y * h))
                    shoulder_px = (int(shoulder_center_x * w), int(shoulder_center_y * h))
                    cv2.circle(frame, nose_px, 8, (0, 255, 255), -1)
                    cv2.circle(frame, shoulder_px, 8, (255, 255, 0), -1)
                    cv2.line(frame, nose_px, shoulder_px, (255, 255, 255), 2)

            # 表示
            if alert_active:
                cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 255), -1)
                text_color = (255, 255, 255)
            else:
                cv2.rectangle(frame, (0, 0), (w, 90), (0, 120, 0), -1)
                text_color = (255, 255, 255)

            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            cv2.putText(frame, f"Score: {score}  {reason}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
            cv2.putText(frame, "Press q to quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Webcam Posture Alert", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
