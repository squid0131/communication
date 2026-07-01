import cv2
import time
import mediapipe as mp

# =============================
# AI Posture Monitor
# Webカメラで姿勢を検出し、姿勢が悪いと画面をぼかす
# q キーで終了
# =============================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 調整しやすい設定
BAD_SECONDS_THRESHOLD = 2.0       # 悪い姿勢が何秒続いたら強いぼかしにするか
HEAD_OFFSET_LIMIT = 0.12          # 顔が肩の中心からどれくらいズレたら大きく減点するか
SHOULDER_TILT_LIMIT = 0.09        # 肩の左右差がどれくらいなら大きく減点するか

bad_start_time = None


def point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return lm.x, lm.y, lm.visibility


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def calculate_score(head_offset, shoulder_tilt):
    """
    点数は100点からの減点式。
    - 顔が肩の中心からズレるほど減点
    - 左右の肩の高さがズレるほど減点
    """
    head_penalty = int((head_offset / HEAD_OFFSET_LIMIT) * 45)
    shoulder_penalty = int((shoulder_tilt / SHOULDER_TILT_LIMIT) * 45)
    score = 100 - head_penalty - shoulder_penalty
    return clamp(score, 0, 100)


def blur_strength_from_score(score, alert_active):
    """
    点数が低いほどぼかしを強くする。
    OpenCVのGaussianBlurは奇数のカーネルサイズが必要。
    """
    if score >= 80:
        return 0
    if score >= 60:
        blur = 9
    elif score >= 40:
        blur = 19
    else:
        blur = 31

    if alert_active:
        blur += 18

    if blur % 2 == 0:
        blur += 1
    return blur


def draw_hud(frame, score, status, elapsed):
    h, w, _ = frame.shape

    # 半透明の黒いパネル
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    cv2.putText(frame, f"Posture Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, status, (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if elapsed > 0:
        cv2.putText(frame, f"bad posture: {elapsed:.1f}s", (w - 290, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(frame, "Press q to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    global bad_start_time

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

            score = 100
            status = "No person detected"
            elapsed = 0
            is_bad = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                nose_x, nose_y, nose_v = point(landmarks, "NOSE")
                ls_x, ls_y, ls_v = point(landmarks, "LEFT_SHOULDER")
                rs_x, rs_y, rs_v = point(landmarks, "RIGHT_SHOULDER")

                if min(nose_v, ls_v, rs_v) > 0.5:
                    shoulder_center_x = (ls_x + rs_x) / 2
                    shoulder_center_y = (ls_y + rs_y) / 2

                    # 正面カメラ想定：顔の中心ズレと肩の傾きから判定
                    head_offset = abs(nose_x - shoulder_center_x)
                    shoulder_tilt = abs(ls_y - rs_y)

                    score = calculate_score(head_offset, shoulder_tilt)
                    is_bad = score < 70

                    if is_bad:
                        if bad_start_time is None:
                            bad_start_time = time.time()
                        elapsed = time.time() - bad_start_time
                    else:
                        bad_start_time = None
                        elapsed = 0

                    if score >= 80:
                        status = "Good posture"
                    elif score >= 60:
                        status = "Posture is getting worse"
                    else:
                        status = "Screen is blurred because posture is poor"

                    # 骨格描画
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    # 鼻と肩中心を表示
                    nose_px = (int(nose_x * w), int(nose_y * h))
                    shoulder_px = (int(shoulder_center_x * w), int(shoulder_center_y * h))
                    cv2.circle(frame, nose_px, 7, (0, 255, 255), -1)
                    cv2.circle(frame, shoulder_px, 7, (255, 255, 0), -1)
                    cv2.line(frame, nose_px, shoulder_px, (255, 255, 255), 2)

            alert_active = elapsed >= BAD_SECONDS_THRESHOLD
            blur = blur_strength_from_score(score, alert_active)

            # 画面全体をぼかす。ただしHUDは後から描くので文字は読める。
            if blur > 0:
                frame = cv2.GaussianBlur(frame, (blur, blur), 0)

            draw_hud(frame, score, status, elapsed)

            cv2.imshow("AI Posture Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
