"""
Tool to test face recognition on a single image.
"""

import os
import sys
import logging
import argparse
import cv2
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.access_control.verifier import AccessVerifier
from src.utils.image import read_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, 'test_recognition.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_recognition(image_path, use_faiss=False, detection_threshold=None, recognition_threshold=None, display=True, save_path=None):
    """
    Test face recognition on a single image.
    
    Args:
        image_path: Path to test image
        use_faiss: Whether to use FAISS for database
        detection_threshold: Detection confidence threshold
        recognition_threshold: Recognition similarity threshold
        display: Whether to display the result
        save_path: Path to save the result image
    
    Returns:
        Recognition result (name, score)
    """
    # Set thresholds
    if detection_threshold is None:
        detection_threshold = settings.DETECTION_THRESHOLD
    
    if recognition_threshold is None:
        recognition_threshold = settings.RECOGNITION_THRESHOLD
    
    # Initialize verifier
    verifier = AccessVerifier(
        detection_threshold=detection_threshold,
        recognition_threshold=recognition_threshold,
        use_faiss=use_faiss
    )
    
    # Read image
    img = read_image(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return None, 0
    
    # Start time
    start_time = time.time()
    
    # Verify face
    name, score, face, all_scores = verifier.verify_face(img, return_details=True)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log result
    if name:
        logger.info(f"Recognized as {name} with score {score:.4f} in {processing_time:.3f}s")
    else:
        logger.info(f"Unknown face with score {score:.4f} in {processing_time:.3f}s")
    
    # Display top matches
    if all_scores:
        logger.info("Top matches:")
        for person, similarity in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {person}: {similarity:.4f}")
    
    # Create annotated image
    if display or save_path:
        result_img = verifier.verify_and_display(img)
        
        # Display result
        if display:
            try:
                cv2.imshow("Recognition Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                logger.warning(f"Could not display image: {e}")
                logger.info("GUI support not available in this OpenCV build.")
                # Save the image instead as fallback
                if save_path is None:
                    fallback_path = os.path.join(settings.DATA_DIR, "results", 
                                                f"result_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                    cv2.imwrite(fallback_path, result_img)
                    logger.info(f"Saved result to {fallback_path} instead")
        
        # Save result
        if save_path:
            cv2.imwrite(save_path, result_img)
            logger.info(f"Saved result to {save_path}")
    
    return name, score

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test face recognition')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--use-faiss', action='store_true',
                        help='Use FAISS for database')
    parser.add_argument('--detection-threshold', type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--recognition-threshold', type=float,
                        help='Recognition similarity threshold')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the result')
    parser.add_argument('--save', type=str,
                        help='Path to save the result image')
    
    args = parser.parse_args()
    
    # Test recognition
    test_recognition(
        image_path=args.image,
        use_faiss=args.use_faiss,
        detection_threshold=args.detection_threshold,
        recognition_threshold=args.recognition_threshold,
        display=not args.no_display,
        save_path=args.save
    )

if __name__ == "__main__":
    main()