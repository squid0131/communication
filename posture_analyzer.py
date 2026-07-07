import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PostureResult:
    score: float
    blur_strength: float
    mode: str
    reasons: List[str]
    detected: bool


class PostureAnalyzer:
    """
    MediaPipe Pose を使って上半身姿勢を評価するクラス。

    評価対象:
    - 過度な覗き込み: 顔サイズが基準より大きくなる
    - 正面の猫背: 鼻が肩中心から下がる/前に寄る、肩の傾き
    - 横向きの猫背: 耳・肩・腰の縦ラインの崩れ
    - 頭の下がり
    """

    def __init__(self, calibration_seconds: float = 5.0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.calibration_seconds = calibration_seconds
        self.calibration_start = time.time()
        self.calibrating = True
        self.samples = []

        self.baseline = {
            "face_size": None,
            "shoulder_width": None,
            "nose_to_shoulder_y": None,
            "ear_shoulder_x_ratio": None,
            "ear_shoulder_hip_angle": None,
        }

    def close(self):
        self.pose.close()

    @staticmethod
    def _point(landmarks, idx, w, h, min_visibility=0.45) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        if lm.visibility < min_visibility:
            return None
        return (lm.x * w, lm.y * h)

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _mid(a, b) -> Tuple[float, float]:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    @staticmethod
    def _angle(a, b, c) -> float:
        """
        点 a-b-c の角度を度で返す。
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        n1 = math.hypot(*ba)
        n2 = math.hypot(*bc)
        if n1 == 0 or n2 == 0:
            return 180.0
        cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cosv))

    def _extract_features(self, frame) -> Optional[Dict[str, float]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        P = self.mp_pose.PoseLandmark

        nose = self._point(lm, P.NOSE.value, w, h)
        left_eye = self._point(lm, P.LEFT_EYE.value, w, h)
        right_eye = self._point(lm, P.RIGHT_EYE.value, w, h)
        left_ear = self._point(lm, P.LEFT_EAR.value, w, h)
        right_ear = self._point(lm, P.RIGHT_EAR.value, w, h)
        left_shoulder = self._point(lm, P.LEFT_SHOULDER.value, w, h)
        right_shoulder = self._point(lm, P.RIGHT_SHOULDER.value, w, h)
        left_hip = self._point(lm, P.LEFT_HIP.value, w, h, min_visibility=0.35)
        right_hip = self._point(lm, P.RIGHT_HIP.value, w, h, min_visibility=0.35)

        if nose is None or left_shoulder is None or right_shoulder is None:
            return None

        shoulder_center = self._mid(left_shoulder, right_shoulder)
        shoulder_width = self._dist(left_shoulder, right_shoulder)

        face_candidates = []
        if left_eye and right_eye:
            face_candidates.append(self._dist(left_eye, right_eye) * 2.4)
        if left_ear and right_ear:
            face_candidates.append(self._dist(left_ear, right_ear))
        if nose and left_ear:
            face_candidates.append(self._dist(nose, left_ear) * 1.8)
        if nose and right_ear:
            face_candidates.append(self._dist(nose, right_ear) * 1.8)
        face_size = max(face_candidates) if face_candidates else shoulder_width * 0.45

        # 横向き判定: 片耳だけ見えやすい/両肩幅が狭く見える/顔の左右が非対称
        ears_visible = int(left_ear is not None) + int(right_ear is not None)
        shoulder_face_ratio = shoulder_width / max(face_size, 1.0)
        side_score = 0.0
        if ears_visible == 1:
            side_score += 1.0
        if shoulder_face_ratio < 2.1:
            side_score += 1.0

        visible_ear = left_ear or right_ear
        visible_shoulder = left_shoulder if left_ear else right_shoulder
        visible_hip = left_hip if left_ear else right_hip

        ear_shoulder_x_ratio = 0.0
        ear_shoulder_hip_angle = 180.0
        if visible_ear and visible_shoulder:
            ear_shoulder_x_ratio = abs(visible_ear[0] - visible_shoulder[0]) / max(shoulder_width, 1.0)
        if visible_ear and visible_shoulder and visible_hip:
            ear_shoulder_hip_angle = self._angle(visible_ear, visible_shoulder, visible_hip)

        mode = "side" if side_score >= 1.5 else "front"

        return {
            "mode": mode,
            "face_size": face_size,
            "shoulder_width": shoulder_width,
            "nose_to_shoulder_y": (nose[1] - shoulder_center[1]) / max(shoulder_width, 1.0),
            "nose_to_shoulder_x": (nose[0] - shoulder_center[0]) / max(shoulder_width, 1.0),
            "shoulder_tilt": abs(left_shoulder[1] - right_shoulder[1]) / max(shoulder_width, 1.0),
            "ear_shoulder_x_ratio": ear_shoulder_x_ratio,
            "ear_shoulder_hip_angle": ear_shoulder_hip_angle,
        }

    def _update_calibration(self, features: Dict[str, float]):
        self.samples.append(features)
        if time.time() - self.calibration_start < self.calibration_seconds:
            return

        def median_value(key):
            values = [s[key] for s in self.samples if key in s and s[key] is not None]
            return float(np.median(values)) if values else None

        self.baseline["face_size"] = median_value("face_size")
        self.baseline["shoulder_width"] = median_value("shoulder_width")
        self.baseline["nose_to_shoulder_y"] = median_value("nose_to_shoulder_y")
        self.baseline["ear_shoulder_x_ratio"] = median_value("ear_shoulder_x_ratio")
        self.baseline["ear_shoulder_hip_angle"] = median_value("ear_shoulder_hip_angle")
        self.calibrating = False

    def analyze(self, frame) -> PostureResult:
        features = self._extract_features(frame)
        if features is None:
            return PostureResult(
                score=75,
                blur_strength=0.0,
                mode="not detected",
                reasons=["上半身が検出できません"],
                detected=False,
            )

        if self.calibrating:
            self._update_calibration(features)
            return PostureResult(
                score=100,
                blur_strength=0.0,
                mode="calibrating",
                reasons=["良い姿勢を登録中"],
                detected=True,
            )

        score = 100.0
        reasons = []

        face_base = self.baseline["face_size"] or features["face_size"]
        face_ratio = features["face_size"] / max(face_base, 1.0)

        # 過度な覗き込み: 顔が基準より大きく見える
        if face_ratio > 1.28:
            penalty = min(35, (face_ratio - 1.28) * 90)
            score -= penalty
            reasons.append("画面に近づきすぎています")

        # 肩の左右傾き
        if features["shoulder_tilt"] > 0.12:
            penalty = min(18, (features["shoulder_tilt"] - 0.12) * 90)
            score -= penalty
            reasons.append("肩が傾いています")

        mode = features["mode"]

        if mode == "front":
            base_nose_y = self.baseline["nose_to_shoulder_y"] or features["nose_to_shoulder_y"]
            head_drop = features["nose_to_shoulder_y"] - base_nose_y

            # 頭が下がる/前のめりの代替指標
            if head_drop > 0.12:
                penalty = min(28, head_drop * 120)
                score -= penalty
                reasons.append("頭が下がっています")

            # 正面で鼻が肩中心から大きくずれる
            if abs(features["nose_to_shoulder_x"]) > 0.28:
                penalty = min(18, (abs(features["nose_to_shoulder_x"]) - 0.28) * 60)
                score -= penalty
                reasons.append("首が横にずれています")

        else:
            # 横向きでは耳・肩・腰の角度、耳と肩のずれを見る
            base_angle = self.baseline["ear_shoulder_hip_angle"] or 170.0
            angle_drop = max(0.0, base_angle - features["ear_shoulder_hip_angle"])

            if angle_drop > 10:
                penalty = min(35, (angle_drop - 10) * 1.4)
                score -= penalty
                reasons.append("横向きで背中が丸まっています")

            base_x = self.baseline["ear_shoulder_x_ratio"] or 0.0
            extra_forward = features["ear_shoulder_x_ratio"] - base_x
            if extra_forward > 0.20:
                penalty = min(28, (extra_forward - 0.20) * 70)
                score -= penalty
                reasons.append("首が前に出ています")

        score = max(0.0, min(100.0, score))

        # 90点以上はぼかしなし、60点以下はほぼ見えない状態。90〜60点の間は線形にぼかす。
        if score >= 90:
            blur_strength = 0.0
        elif score <= 60:
            blur_strength = 1.0
        else:
            blur_strength = (90 - score) / 30

        if not reasons:
            reasons.append("姿勢は良好です")

        return PostureResult(
            score=score,
            blur_strength=blur_strength,
            mode=mode,
            reasons=reasons,
            detected=True,
        )
