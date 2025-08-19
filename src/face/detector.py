"""
Face detection module using InsightFace.
"""

import logging
import numpy as np
from insightface.app import FaceAnalysis
import cv2

from config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector class using InsightFace."""

    def __init__(
        self,
        model_name=settings.MODEL_NAME,
        det_size=settings.DET_SIZE,
        ctx_id=settings.CTX_ID,
        detection_threshold=settings.DETECTION_THRESHOLD,
    ):
        """
        Initialize face detector.

        Args:
            model_name: InsightFace model name
            det_size: Detection size (width, height)
            ctx_id: Context ID (0 for GPU, -1 for CPU)
            detection_threshold: Confidence threshold for face detection
        """
        self.model_name = model_name
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.detection_threshold = detection_threshold
        self.app = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the InsightFace model."""
        try:
            logger.info(f"Initializing face detector with model: {self.model_name}")
            logger.info(
                f"Using context ID: {self.ctx_id} ({'GPU' if self.ctx_id >= 0 else 'CPU'})"
            )
            logger.info(f"Detection size: {self.det_size}")

            # Use only CPUExecutionProvider when ctx_id is -1 (CPU mode)
            if self.ctx_id == -1:
                providers = ["CPUExecutionProvider"]
                logger.info("Using CPU-only execution")
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Using GPU with CPU fallback")

            self.app = FaceAnalysis(name=self.model_name, providers=providers)
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            logger.info(
                f"Successfully initialized face detector with model: {self.model_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise

    def detect_faces(self, img):
        """
        Detect faces in an image.

        Args:
            img: Input image (numpy array)

        Returns:
            List of detected faces from InsightFace
        """
        if img is None:
            logger.warning("Empty image provided to detect_faces")
            return []

        try:
            faces = self.app.get(img)
            logger.debug(f"Detected {len(faces)} faces")
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def get_largest_face(self, img):
        """
        Get the largest face in an image.

        Args:
            img: Input image (numpy array)

        Returns:
            Largest face or None if no faces detected
        """
        faces = self.detect_faces(img)

        if not faces:
            return None

        # Get largest face by bounding box area
        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1]),
        )

        # Filter by detection threshold
        if largest_face.det_score < self.detection_threshold:
            logger.debug(
                f"Largest face detection score {largest_face.det_score} below threshold"
            )
            return None

        return largest_face

    def get_face_locations(self, face):
        """
        Get face bounding box locations.

        Args:
            face: Face object from InsightFace

        Returns:
            tuple: (top, right, bottom, left) coordinates
        """
        if face is None:
            return None

        bbox = face.bbox.astype(int)
        return (bbox[1], bbox[2], bbox[3], bbox[0])  # top, right, bottom, left

    def draw_face_locations(self, img, face, name=None, score=None):
        """
        Draw face bounding boxes and information on image.

        Args:
            img: Input image (numpy array)
            face: Face object from InsightFace
            name: Name of the person (optional)
            score: Recognition score (optional)

        Returns:
            Image with annotations
        """
        if face is None:
            return img

        # Create a copy of the image
        annotated_img = img.copy()

        # Get bbox
        bbox = face.bbox.astype(int)

        # Determine color based on recognition
        if name:
            color = (0, 255, 0)  # Green for recognized
            text = f"{name}: {score:.2f}" if score else name
        else:
            color = (0, 0, 255)  # Red for unknown
            text = "Unknown"

        # Draw rectangle
        cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw text
        cv2.putText(
            annotated_img,
            text,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        return annotated_img
