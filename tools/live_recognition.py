"""
Tool to run live face recognition using webcam.
"""

import os
import sys
import logging
import argparse
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.access_control.verifier import AccessVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, 'live_recognition.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_live_recognition(camera_id=0, use_faiss=False, detection_threshold=None, recognition_threshold=None, record=False, output_path=None):
    """
    Run live face recognition using webcam.
    
    Args:
        camera_id: Camera ID
        use_faiss: Whether to use FAISS for database
        detection_threshold: Detection confidence threshold
        recognition_threshold: Recognition similarity threshold
        record: Whether to record video
        output_path: Path to save recorded video
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
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    
    # Create video writer if recording
    video_writer = None
    if record:
        if output_path is None:
            output_path = os.path.join(settings.DATA_DIR, "recordings", 
                                       f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, 20.0, (frame_width, frame_height))
        
        logger.info(f"Recording video to {output_path}")
    
    logger.info(f"Starting live recognition with camera {camera_id}")
    logger.info(f"Detection threshold: {detection_threshold}")
    logger.info(f"Recognition threshold: {recognition_threshold}")
    logger.info(f"Using {'FAISS' if use_faiss else 'standard'} database")
    logger.info("Press 'q' to quit, 's' to take screenshot")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Verify face and create annotated image
            result = verifier.verify_and_display(frame)
            
            # Write frame if recording
            if video_writer is not None:
                video_writer.write(result)
            
            # Display frame
            try:
                cv2.imshow("Face Recognition", result)
            except cv2.error as e:
                logger.error(f"Could not display frame: {e}")
                logger.info("GUI support not available in this OpenCV build.")
                # Save some frames as fallback if recording is not enabled
                if not record and int(time.time()) % 5 == 0:  # Save every 5 seconds
                    fallback_path = os.path.join(settings.DATA_DIR, "frames", 
                                               f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                    cv2.imwrite(fallback_path, result)
                break  # Exit the loop as we can't display frames
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Quit if 'q' pressed
            if key == ord('q'):
                break
            
            # Take screenshot if 's' pressed
            elif key == ord('s'):
                screenshot_path = os.path.join(settings.DATA_DIR, "screenshots", 
                                              f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                
                # Save screenshot
                cv2.imwrite(screenshot_path, result)
                logger.info(f"Saved screenshot to {screenshot_path}")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
    
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Log access attempts
        access_log = verifier.get_access_log()
        logger.info(f"Total access attempts: {len(access_log)}")
        
        # Count granted vs denied
        granted = sum(1 for entry in access_log if entry['access_granted'])
        denied = len(access_log) - granted
        
        logger.info(f"Access granted: {granted}, denied: {denied}")
        logger.info("Live recognition stopped")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run live face recognition')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--use-faiss', action='store_true',
                        help='Use FAISS for database')
    parser.add_argument('--detection-threshold', type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--recognition-threshold', type=float,
                        help='Recognition similarity threshold')
    parser.add_argument('--record', action='store_true',
                        help='Record video')
    parser.add_argument('--output', type=str,
                        help='Path to save recorded video')
    
    args = parser.parse_args()
    
    # Run live recognition
    run_live_recognition(
        camera_id=args.camera,
        use_faiss=args.use_faiss,
        detection_threshold=args.detection_threshold,
        recognition_threshold=args.recognition_threshold,
        record=args.record,
        output_path=args.output
    )

if __name__ == "__main__":
    main()