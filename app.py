import time
import cv2

from posture_analyzer import PostureAnalyzer
from screen_blur_overlay import ScreenBlurOverlay


def smooth(current, target, up=0.10, down=0.18):
    """
    ブラー強度を急に変えず、徐々に変化させる。
    """
    if target > current:
        return current + (target - current) * up
    return current + (target - current) * down


def main():
    print("Posture Background Blur Advanced")
    print("--------------------------------")
    print("起動しました。")
    print("最初の5秒間は、良い姿勢で座ってください。")
    print("カメラ映像は表示されません。")
    print("終了: Ctrl + C")
    print()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Webカメラを開けませんでした。")
        print("カメラが接続されているか、他のアプリが使用していないか確認してください。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    analyzer = PostureAnalyzer(calibration_seconds=5.0)
    overlay = ScreenBlurOverlay()
    overlay.start()

    blur = 0.0
    last_log = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            # 鏡のような左右反転。姿勢評価の違和感を減らす。
            frame = cv2.flip(frame, 1)

            result = analyzer.analyze(frame)
            blur = smooth(blur, result.blur_strength)
            overlay.set_strength(blur)

            now = time.time()
            if now - last_log > 1.0:
                status = " / ".join(result.reasons[:2])
                print(
                    f"score={result.score:5.1f} "
                    f"blur={blur:4.2f} "
                    f"mode={result.mode:12s} "
                    f"{status}"
                )
                last_log = now

            # CPU負荷を下げるため少し待つ
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        overlay.stop()
        analyzer.close()
        cap.release()


if __name__ == "__main__":
    main()
