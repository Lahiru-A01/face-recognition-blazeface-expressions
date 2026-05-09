import cv2
import mediapipe as mp

class BlazeFaceDetector:
    def __init__(self, min_conf=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=min_conf
        )

    def detect(self, frame):
        """Processes the frame and returns detections."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.detector.process(rgb_frame)