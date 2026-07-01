import threading
import time
import tkinter as tk

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk


class ScreenBlurOverlay:
    """
    画面全体に、スクリーンショットをぼかしたオーバーレイを表示する。

    実際にWindowsのデスクトップ描画を変更しているわけではなく、
    画面全体を覆う最前面ウィンドウに、ぼかしたスクリーンショットを表示する方式。
    """

    def __init__(self):
        self.root = None
        self.label = None
        self.running = False
        self.current_strength = 0.0
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_tk, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.root:
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                pass

    def set_strength(self, strength: float):
        self.current_strength = max(0.0, min(1.0, float(strength)))

    def _run_tk(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)

        # ESCで終了
        self.root.bind("<Escape>", lambda e: self.stop())

        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)

        self._update_frame()
        self.root.mainloop()

    def _update_frame(self):
        if not self.running or not self.root:
            return

        strength = self.current_strength

        if strength <= 0.02:
            # ほぼぼかし不要なら透明気味にする
            try:
                self.root.attributes("-alpha", 0.01)
            except Exception:
                pass
        else:
            try:
                self.root.attributes("-alpha", 0.92)
            except Exception:
                pass

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                img = np.array(sct.grab(monitor))

            # BGRA -> BGR
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # strengthが大きいほどカーネルを大きくする
            k = int(9 + strength * 55)
            if k % 2 == 0:
                k += 1
            blurred = cv2.GaussianBlur(frame, (k, k), 0)

            # 少し白っぽい曇りを重ねる
            haze = np.full_like(blurred, 255)
            blurred = cv2.addWeighted(blurred, 1.0 - 0.22 * strength, haze, 0.22 * strength, 0)

            rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)

            self.label.configure(image=photo)
            self.label.image = photo

        self.root.after(120, self._update_frame)
