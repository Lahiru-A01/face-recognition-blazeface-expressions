import cv2
import numpy as np

def align_face(image, detection):
    """Straightens the face based on eye landmarks."""
    if not detection.location_data.relative_keypoints:
        return image

    landmarks = detection.location_data.relative_keypoints
    h, w, _ = image.shape
    
    # Landmark 0: Left Eye | Landmark 1: Right Eye
    left_eye = (int(landmarks[0].x * w), int(landmarks[0].y * h))
    right_eye = (int(landmarks[1].x * w), int(landmarks[1].y * h))

    # Calculate Angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Rotation Matrix
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    
    return cv2.warpAffine(image, M, (w, h))