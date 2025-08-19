"""
Main entry point for the face recognition access control system.
This script provides a simple command-line interface to the system.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

from config import settings
from src.access_control.verifier import AccessVerifier
from tools.create_database import create_database_from_folders
from tools.test_recognition import test_recognition
from tools.live_recognition import run_live_recognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, f'main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Face Recognition Access Control System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create database from folders
  python main.py create-db --root /path/to/face/data
  
  # Test recognition with an image
  python main.py test --image /path/to/test.jpg
  
  # Run live recognition
  python main.py live
  
  # Run live recognition with recording
  python main.py live --record
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create database parser
    create_parser = subparsers.add_parser('create-db', help='Create face embeddings database')
    create_parser.add_argument('--root', type=str, required=True,
                               help='Root folder containing person subfolders')
    create_parser.add_argument('--db-path', type=str,
                               help='Path to store the database')
    create_parser.add_argument('--use-faiss', action='store_true',
                               help='Use FAISS for database')
    create_parser.add_argument('--min-faces', type=int, default=1,
                               help='Minimum number of faces required per person')
    
    # Test recognition parser
    test_parser = subparsers.add_parser('test', help='Test face recognition on a single image')
    test_parser.add_argument('--image', type=str, required=True,
                             help='Path to test image')
    test_parser.add_argument('--use-faiss', action='store_true',
                             help='Use FAISS for database')
    test_parser.add_argument('--detection-threshold', type=float,
                             help='Detection confidence threshold')
    test_parser.add_argument('--recognition-threshold', type=float,
                             help='Recognition similarity threshold')
    test_parser.add_argument('--no-display', action='store_true',
                             help='Do not display the result')
    test_parser.add_argument('--save', type=str,
                             help='Path to save the result image')
    
    # Live recognition parser
    live_parser = subparsers.add_parser('live', help='Run live face recognition')
    live_parser.add_argument('--camera', type=int, default=0,
                             help='Camera ID')
    live_parser.add_argument('--use-faiss', action='store_true',
                             help='Use FAISS for database')
    live_parser.add_argument('--detection-threshold', type=float,
                             help='Detection confidence threshold')
    live_parser.add_argument('--recognition-threshold', type=float,
                             help='Recognition similarity threshold')
    live_parser.add_argument('--record', action='store_true',
                             help='Record video')
    live_parser.add_argument('--output', type=str,
                             help='Path to save recorded video')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command is provided
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'create-db':
        logger.info("Creating database...")
        create_database_from_folders(
            root_folder=args.root,
            db_path=args.db_path,
            use_faiss=args.use_faiss,
            min_faces_per_person=args.min_faces
        )
        logger.info("Database creation completed")
    
    elif args.command == 'test':
        logger.info("Testing recognition...")
        test_recognition(
            image_path=args.image,
            use_faiss=args.use_faiss,
            detection_threshold=args.detection_threshold,
            recognition_threshold=args.recognition_threshold,
            display=not args.no_display,
            save_path=args.save
        )
        logger.info("Testing completed")
    
    elif args.command == 'live':
        logger.info("Starting live recognition...")
        run_live_recognition(
            camera_id=args.camera,
            use_faiss=args.use_faiss,
            detection_threshold=args.detection_threshold,
            recognition_threshold=args.recognition_threshold,
            record=args.record,
            output_path=args.output
        )
        logger.info("Live recognition completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        sys.exit(1)