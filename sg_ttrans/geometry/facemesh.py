"""
MediaPipe FaceMesh wrapper for extracting 68 canonical 3D facial landmarks.
"""

import cv2
import numpy as np

# Canonical 68 landmark index mapping from MediaPipe FaceMesh (468 landmarks)
# Corresponding to standard 68 anatomical points:
# Jaw (17), Right Eyebrow (5), Left Eyebrow (5), Nose (9), Right Eye (6), Left Eye (6), Outer Lips (12), Inner Lips (8)
FACEMESH_68_INDICES = [
    # Jaw line: 0-16
    234, 93, 132, 58, 172, 136, 150, 149, 152, 377, 400, 378, 379, 365, 397, 288, 454,
    # Right eyebrow: 17-21
    70, 63, 105, 66, 107,
    # Left eyebrow: 22-26
    336, 296, 334, 293, 300,
    # Nose: 27-35
    168, 6, 197, 195, 5, 4, 1, 19, 94,
    # Left eye: 36-41
    33, 160, 158, 133, 153, 144,
    # Right eye: 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lips: 48-59
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    # Inner lips: 60-67 (p12, p13, p14, p15, p16, p17, p18 in paper)
    78, 191, 80, 81, 82, 13, 312, 311
]

class FaceMeshExtractor:
    """
    Extracts 68 anatomical 3D landmark points using MediaPipe FaceMesh.
    """
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.face_mesh = None
        self._init_mediapipe(
            static_image_mode, max_num_faces, min_detection_confidence, min_tracking_confidence
        )

    def _init_mediapipe(self, static_mode: bool, max_faces: int, min_det: float, min_track: float):
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=static_mode,
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_det,
                    min_tracking_confidence=min_track
                )
        except Exception:
            self.face_mesh = None

    def extract(self, bgr_image: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """
        Extracts 68 canonical landmarks from BGR frame.
        Returns:
            tuple[np.ndarray | None, bool]: (landmarks array of shape (68, 3), success_flag)
        """
        if self.face_mesh is None or bgr_image is None or bgr_image.size == 0:
            return None, False

        h, w, _ = bgr_image.shape
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None, False

        face_landmarks = results.multi_face_landmarks[0]
        raw_pts = face_landmarks.landmark

        # Sample the 68 canonical landmark positions
        landmarks_68 = np.zeros((68, 3), dtype=np.float32)
        for i, idx in enumerate(FACEMESH_68_INDICES[:68]):
            if idx < len(raw_pts):
                pt = raw_pts[idx]
                landmarks_68[i] = [pt.x * w, pt.y * h, pt.z * w]
            else:
                landmarks_68[i] = [0.0, 0.0, 0.0]

        return landmarks_68, True

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
