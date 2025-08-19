"""
Face verification module for access control.
"""

import logging
import time
import cv2
import numpy as np
from datetime import datetime

from config import settings
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.database.embeddings_db import EmbeddingsDatabase
from src.database.faiss_db import FaissDatabase
from src.utils.image import draw_access_status

logger = logging.getLogger(__name__)

class AccessVerifier:
    """Face verification for access control."""
    
    def __init__(self, 
                 detection_threshold=settings.DETECTION_THRESHOLD,
                 recognition_threshold=settings.RECOGNITION_THRESHOLD,
                 use_faiss=settings.USE_FAISS):
        """
        Initialize access verifier.
        
        Args:
            detection_threshold: Confidence threshold for face detection
            recognition_threshold: Similarity threshold for face recognition
            use_faiss: Whether to use FAISS for database operations
        """
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.use_faiss = use_faiss
        
        # Initialize components
        self.detector = FaceDetector(detection_threshold=detection_threshold)
        self.embedder = FaceEmbedder()
        
        # Initialize database
        if use_faiss:
            self.database = FaissDatabase()
        else:
            self.database = EmbeddingsDatabase()
        
        logger.info(f"Initialized access verifier with {'FAISS' if use_faiss else 'standard'} database")
        
        # Access log
        self.access_log = []
    
    def verify_face(self, img, return_details=False):
        """
        Verify if the face in the image matches anyone in the database.
        
        Args:
            img: Input image containing a face
            return_details: Whether to return detailed matching information
            
        Returns:
            If return_details is False:
                (person_name, confidence) if match found, (None, 0) otherwise
            If return_details is True:
                (person_name, confidence, face_object, all_scores) if match found,
                (None, 0, face_object, all_scores) otherwise
        """
        # Detect face
        face = self.detector.get_largest_face(img)
        
        # No face detected
        if face is None:
            if return_details:
                return None, 0, None, {}
            return None, 0
        
        # Get embedding
        embedding = self.embedder.get_embedding(face)
        
        if embedding is None:
            if return_details:
                return None, 0, face, {}
            return None, 0
        
        # Match against database
        name, score = self.database.find_match(embedding, self.recognition_threshold)
        
        # Get all scores if requested
        all_scores = {}
        if return_details:
            all_scores = self.database.get_all_similarity_scores(embedding)
        
        # Log access attempt
        self._log_access_attempt(name, score, face)
        
        if return_details:
            return name, score, face, all_scores
        return name, score
    
    def _log_access_attempt(self, name, score, face):
        """
        Log access attempt.
        
        Args:
            name: Person name or None
            score: Recognition score
            face: Face object
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'person_name': name if name else "Unknown",
            'score': score,
            'access_granted': name is not None,
            'detection_score': face.det_score if face else 0
        }
        
        # Add to log
        self.access_log.append(log_entry)
        
        # Log access attempt
        status = "GRANTED" if name else "DENIED"
        logger.info(f"Access {status} - Person: {name if name else 'Unknown'}, Score: {score:.4f}")
    
    def verify_and_display(self, img, display_scores=True):
        """
        Verify face and create annotated image for display.
        
        Args:
            img: Input image
            display_scores: Whether to display scores
        
        Returns:
            Annotated image
        """
        if img is None:
            return None
        
        # Make a copy of the image
        result = img.copy()
        
        # Verify face
        name, score, face, all_scores = self.verify_face(img, return_details=True)
        
        # Draw face bounding box if face detected
        if face is not None:
            # Draw bounding box
            result = self.detector.draw_face_locations(result, face, name, score)
            
            # Draw access status
            result = draw_access_status(result, granted=name is not None)
            
            # Draw scores if requested
            if display_scores and all_scores:
                # Sort scores in descending order
                sorted_scores = sorted(all_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)
                
                # Display top 3 scores
                y_pos = 100
                for i, (person, similarity) in enumerate(sorted_scores[:3]):
                    text = f"{i+1}. {person}: {similarity:.4f}"
                    cv2.putText(result, text, 
                               (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 0), 2)
                    y_pos += 30
        
        return result
    
    def run_live_verification(self, camera_id=0, window_name="Face Verification"):
        """
        Run live face verification using webcam.
        
        Args:
            camera_id: Camera ID
            window_name: Window name
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        
        logger.info(f"Starting live verification. Press 'q' to quit.")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Error: Could not read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Start time
                start_time = time.time()
                
                # Verify face and create annotated image
                result = self.verify_and_display(frame)
                
                # Calculate FPS
                fps = 1 / (time.time() - start_time)
                
                # Display FPS
                cv2.putText(result, f"FPS: {fps:.2f}", 
                           (result.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow(window_name, result)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Live verification stopped")
    
    def get_access_log(self, limit=None):
        """
        Get access log.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of access log entries
        """
        if limit is None:
            return self.access_log
        return self.access_log[-limit:]