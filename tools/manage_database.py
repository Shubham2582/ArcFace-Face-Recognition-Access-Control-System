"""
Database management tool for the face recognition system.
Allows listing, adding, removing, and updating identities.
"""

import os
import sys
import logging
import argparse
import cv2
import numpy as np
import pickle
import time
from datetime import datetime
from prettytable import PrettyTable

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.database.embeddings_db import EmbeddingsDatabase
from src.database.faiss_db import FaissDatabase
from src.utils.image import read_image, is_image_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, 'manage_database.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_database(use_faiss=False, db_path=None):
    """
    Get database instance.
    
    Args:
        use_faiss: Whether to use FAISS
        db_path: Custom database path
    
    Returns:
        Database instance
    """
    if use_faiss:
        if db_path:
            return FaissDatabase(db_path=db_path)
        return FaissDatabase()
    else:
        if db_path:
            return EmbeddingsDatabase(db_path=db_path)
        return EmbeddingsDatabase()

def list_identities(args):
    """List all identities in the database."""
    db = get_database(args.use_faiss, args.db_path)
    identities = db.list_identities()
    
    if not identities:
        print("No identities found in the database.")
        return
    
    table = PrettyTable()
    
    if isinstance(db, EmbeddingsDatabase):
        table.field_names = ["Name", "Images Used", "Created At", "Updated At"]
        for name in identities:
            data = db.get_identity(name)
            table.add_row([
                name, 
                data['num_images'], 
                data.get('created_at', 'N/A'), 
                data.get('updated_at', 'N/A')
            ])
    else:  # FaissDatabase
        table.field_names = ["Name", "Images Used", "Created At", "Updated At", "Index"]
        for name in identities:
            data = db.get_identity_metadata(name)
            if data:
                table.add_row([
                    name, 
                    data.get('num_images', 'N/A'), 
                    data.get('created_at', 'N/A'), 
                    data.get('updated_at', 'N/A'),
                    data.get('index', 'N/A')
                ])
            else:
                table.add_row([name, 'N/A', 'N/A', 'N/A', 'N/A'])
    
    print(table)
    print(f"\nTotal identities: {len(identities)}")

def add_identity(args):
    """Add a new identity to the database."""
    # Initialize components
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = get_database(args.use_faiss, args.db_path)
    
    # Check if identity already exists
    identities = db.list_identities()
    if args.name in identities and not args.force:
        print(f"Identity '{args.name}' already exists. Use --force to overwrite.")
        return
    
    # Process image folder if provided
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Folder not found: {args.folder}")
            return
        
        # Get list of image files
        image_files = [f for f in os.listdir(args.folder) 
                      if is_image_file(os.path.join(args.folder, f))]
        
        if not image_files:
            print(f"No images found in folder: {args.folder}")
            return
        
        embeddings = []
        valid_images = 0
        
        # Process each image
        for img_name in image_files:
            img_path = os.path.join(args.folder, img_name)
            print(f"Processing {img_path}...")
            
            # Read image
            img = read_image(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Detect face
            face = detector.get_largest_face(img)
            if face is None:
                print(f"No face detected in {img_path}")
                continue
            
            # Get embedding
            embedding = embedder.get_embedding(face)
            if embedding is None:
                print(f"Failed to get embedding for {img_path}")
                continue
            
            # Add embedding
            embeddings.append(embedding)
            valid_images += 1
        
        if valid_images == 0:
            print("No valid face images found.")
            return
        
        # Calculate average embedding
        avg_embedding = embedder.average_embeddings(embeddings)
        
        if avg_embedding is None:
            print("Failed to calculate average embedding.")
            return
        
        # Add to database
        db.add_identity(args.name, avg_embedding, valid_images)
        db.save_database()
        
        print(f"Added identity '{args.name}' with {valid_images} images.")
    
    # Process single image if provided
    elif args.image:
        if not os.path.isfile(args.image):
            print(f"Image not found: {args.image}")
            return
        
        # Read image
        img = read_image(args.image)
        if img is None:
            print(f"Could not read image: {args.image}")
            return
        
        # Detect face
        face = detector.get_largest_face(img)
        if face is None:
            print(f"No face detected in {args.image}")
            return
        
        # Get embedding
        embedding = embedder.get_embedding(face)
        if embedding is None:
            print(f"Failed to get embedding for {args.image}")
            return
        
        # Add to database
        db.add_identity(args.name, embedding, 1)
        db.save_database()
        
        print(f"Added identity '{args.name}' with 1 image.")
    
    else:
        print("Error: Either --folder or --image must be provided.")
        return

def remove_identity(args):
    """Remove an identity from the database."""
    db = get_database(args.use_faiss, args.db_path)
    
    # Check if identity exists
    identities = db.list_identities()
    if args.name not in identities:
        print(f"Identity '{args.name}' not found in the database.")
        return
    
    # Confirm removal
    if not args.force:
        confirm = input(f"Are you sure you want to remove '{args.name}'? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Remove identity
    if isinstance(db, EmbeddingsDatabase):
        success = db.remove_identity(args.name)
        if success:
            db.save_database()
            print(f"Removed identity '{args.name}' from the database.")
        else:
            print(f"Failed to remove identity '{args.name}'.")
    else:
        print("FAISS database doesn't support direct removal. Creating a new database without the identity...")
        
        # For FAISS, we need to rebuild the database without the identity
        # This is a limitation of FAISS (it doesn't support removing entries)
        temp_db_path = args.db_path + ".temp" if args.db_path else settings.FAISS_DB_PATH + ".temp"
        temp_labels_path = settings.FAISS_LABELS_PATH + ".temp"
        
        # Create new database
        new_db = FaissDatabase(db_path=temp_db_path, labels_path=temp_labels_path)
        
        # Copy all identities except the one to remove
        for identity in identities:
            if identity != args.name:
                # Get metadata
                metadata = db.get_identity_metadata(identity)
                if metadata and 'index' in metadata:
                    # This is simplified - in a real system, you'd need to get the actual embedding
                    # For now, we just show the concept
                    print(f"Would copy identity '{identity}' to new database.")
        
        print(f"To actually rebuild a FAISS database, you need to re-create it from the original images.")
        print(f"This is because FAISS doesn't store the actual embeddings in a retrievable format for each identity.")

def update_identity(args):
    """Update an identity in the database."""
    # This is similar to add_identity but checking if the identity exists first
    add_identity(args)

def rename_identity(args):
    """Rename an identity in the database."""
    db = get_database(args.use_faiss, args.db_path)
    
    # Check if old identity exists
    identities = db.list_identities()
    if args.old_name not in identities:
        print(f"Identity '{args.old_name}' not found in the database.")
        return
    
    # Check if new identity already exists
    if args.new_name in identities and not args.force:
        print(f"Identity '{args.new_name}' already exists. Use --force to overwrite.")
        return
    
    if isinstance(db, EmbeddingsDatabase):
        # Get the identity data
        identity_data = db.get_identity(args.old_name)
        
        # Add with new name
        db.add_identity(args.new_name, identity_data['embedding'], identity_data['num_images'])
        
        # Remove old identity
        db.remove_identity(args.old_name)
        
        # Save database
        db.save_database()
        
        print(f"Renamed identity from '{args.old_name}' to '{args.new_name}'.")
    else:
        print("FAISS database doesn't support direct renaming. Please use the following workaround:")
        print("1. Add the identity with the new name using the same images")
        print("2. Remove the old identity (which will require rebuilding the database)")

def export_database(args):
    """Export database to a different format."""
    source_db = get_database(args.use_faiss, args.db_path)
    
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"database_export_{timestamp}.pkl"
    
    identities = source_db.list_identities()
    if not identities:
        print("No identities found in the database.")
        return
    
    export_data = {
        "format_version": "1.0",
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "identities": {}
    }
    
    if isinstance(source_db, EmbeddingsDatabase):
        # Standard database - we can directly access the data
        for name in identities:
            data = source_db.get_identity(name)
            export_data["identities"][name] = {
                "embedding": data['embedding'].tolist(),  # Convert numpy array to list for JSON compatibility
                "num_images": data['num_images'],
                "created_at": data.get('created_at', 'N/A'),
                "updated_at": data.get('updated_at', 'N/A')
            }
    else:
        print("FAISS database export is limited to metadata only.")
        print("Full embedding export from FAISS is not supported in this tool.")
        # We could add functionality to export FAISS index directly, but it's complex
        return
    
    # Save export file
    with open(args.output, 'wb') as f:
        pickle.dump(export_data, f)
    
    print(f"Exported {len(identities)} identities to {args.output}")

def import_database(args):
    """Import database from a file."""
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    try:
        with open(args.input, 'rb') as f:
            import_data = pickle.load(f)
        
        if "format_version" not in import_data or "identities" not in import_data:
            print("Invalid import file format.")
            return
        
        target_db = get_database(args.use_faiss, args.db_path)
        
        # Get existing identities
        existing_identities = set(target_db.list_identities())
        import_identities = set(import_data["identities"].keys())
        
        # Check for conflicts
        conflicts = existing_identities.intersection(import_identities)
        if conflicts and not args.force:
            print(f"Found {len(conflicts)} conflicting identities.")
            print("Use --force to overwrite existing identities.")
            print(f"Conflicting identities: {', '.join(conflicts)}")
            return
        
        # Import identities
        for name, data in import_data["identities"].items():
            if "embedding" in data:
                embedding = np.array(data["embedding"])
                num_images = data.get("num_images", 1)
                
                target_db.add_identity(name, embedding, num_images)
        
        target_db.save_database()
        
        print(f"Imported {len(import_data['identities'])} identities to the database.")
        
    except Exception as e:
        print(f"Error importing database: {e}")
        return

def backup_database(args):
    """Create a backup of the database."""
    db = get_database(args.use_faiss, args.db_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.use_faiss:
        source_db_path = args.db_path if args.db_path else settings.FAISS_DB_PATH
        source_labels_path = settings.FAISS_LABELS_PATH
        
        backup_db_path = args.output if args.output else f"{source_db_path}.backup_{timestamp}"
        backup_labels_path = f"{os.path.splitext(backup_db_path)[0]}_labels.pkl"
        
        try:
            # Copy the FAISS index file
            import shutil
            shutil.copy2(source_db_path, backup_db_path)
            shutil.copy2(source_labels_path, backup_labels_path)
            
            print(f"Created backup of FAISS database:")
            print(f"  Index: {backup_db_path}")
            print(f"  Labels: {backup_labels_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
    else:
        source_db_path = args.db_path if args.db_path else settings.DB_PATH
        backup_db_path = args.output if args.output else f"{source_db_path}.backup_{timestamp}"
        
        try:
            # Copy the database file
            import shutil
            shutil.copy2(source_db_path, backup_db_path)
            
            print(f"Created backup of database: {backup_db_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")

def test_identity(args):
    """Test recognition for a specific identity."""
    from src.access_control.verifier import AccessVerifier
    
    if not args.image:
        print("Error: --image is required for testing.")
        return
    
    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        return
    
    # Initialize verifier
    detection_threshold = args.detection_threshold or settings.DETECTION_THRESHOLD
    recognition_threshold = args.recognition_threshold or settings.RECOGNITION_THRESHOLD
    
    verifier = AccessVerifier(
        detection_threshold=detection_threshold,
        recognition_threshold=recognition_threshold,
        use_faiss=args.use_faiss
    )
    
    # Read image
    img = read_image(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        return
    
    # Verify face
    name, score, face, all_scores = verifier.verify_face(img, return_details=True)
    
    if face is None:
        print("No face detected in the image.")
        return
    
    print(f"Detection score: {face.det_score:.4f}")
    
    if name:
        print(f"Recognized as: {name}")
        print(f"Confidence score: {score:.4f}")
    else:
        print("No match found above threshold.")
    
    # Print top 5 matches
    print("\nTop 5 matches:")
    for person, similarity in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {person}: {similarity:.4f}")
    
    # If a specific identity was provided, show its score
    if args.name and args.name in all_scores:
        print(f"\nScore for '{args.name}': {all_scores[args.name]:.4f}")
    
    # Display image if requested
    if not args.no_display:
        try:
            # Create annotated image
            result_img = verifier.verify_and_display(img)
            
            # Display
            cv2.imshow("Test Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image: {e}")
            
            # Save result instead
            result_path = f"test_result_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(result_path, verifier.verify_and_display(img))
            print(f"Saved result to {result_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Face Recognition Database Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all identities
  python manage_database.py list
  
  # Add a new identity from a folder of images
  python manage_database.py add --name "John Doe" --folder /path/to/john_images
  
  # Add a new identity from a single image
  python manage_database.py add --name "Jane Doe" --image /path/to/jane.jpg
  
  # Remove an identity
  python manage_database.py remove --name "John Doe"
  
  # Rename an identity
  python manage_database.py rename --old-name "John Doe" --new-name "Johnny"
  
  # Export database
  python manage_database.py export --output backup.pkl
  
  # Import database
  python manage_database.py import --input backup.pkl
  
  # Create a backup
  python manage_database.py backup
  
  # Test recognition for a specific person
  python manage_database.py test --image /path/to/test.jpg --name "John Doe"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--use-faiss', action='store_true',
                              help='Use FAISS for database operations')
    parent_parser.add_argument('--db-path', type=str,
                              help='Path to the database')
    
    # List command
    list_parser = subparsers.add_parser('list', parents=[parent_parser],
                                      help='List all identities in the database')
    
    # Add command
    add_parser = subparsers.add_parser('add', parents=[parent_parser],
                                      help='Add a new identity')
    add_parser.add_argument('--name', type=str, required=True,
                          help='Name of the identity')
    add_parser.add_argument('--folder', type=str,
                          help='Folder containing images of the identity')
    add_parser.add_argument('--image', type=str,
                          help='Single image of the identity')
    add_parser.add_argument('--force', action='store_true',
                          help='Force overwrite if identity already exists')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', parents=[parent_parser],
                                        help='Remove an identity')
    remove_parser.add_argument('--name', type=str, required=True,
                             help='Name of the identity to remove')
    remove_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    
    # Update command
    update_parser = subparsers.add_parser('update', parents=[parent_parser],
                                        help='Update an identity')
    update_parser.add_argument('--name', type=str, required=True,
                             help='Name of the identity to update')
    update_parser.add_argument('--folder', type=str,
                             help='Folder containing images of the identity')
    update_parser.add_argument('--image', type=str,
                             help='Single image of the identity')
    update_parser.add_argument('--force', action='store_true',
                             help='Force update')
    
    # Rename command
    rename_parser = subparsers.add_parser('rename', parents=[parent_parser],
                                        help='Rename an identity')
    rename_parser.add_argument('--old-name', type=str, required=True,
                             help='Current name of the identity')
    rename_parser.add_argument('--new-name', type=str, required=True,
                             help='New name for the identity')
    rename_parser.add_argument('--force', action='store_true',
                             help='Force overwrite if new name already exists')
    
    # Export command
    export_parser = subparsers.add_parser('export', parents=[parent_parser],
                                        help='Export database to file')
    export_parser.add_argument('--output', type=str,
                             help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', parents=[parent_parser],
                                        help='Import database from file')
    import_parser.add_argument('--input', type=str, required=True,
                             help='Input file path')
    import_parser.add_argument('--force', action='store_true',
                             help='Force overwrite of existing identities')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', parents=[parent_parser],
                                        help='Create a backup of the database')
    backup_parser.add_argument('--output', type=str,
                             help='Backup file path')
    
    # Test command
    test_parser = subparsers.add_parser('test', parents=[parent_parser],
                                      help='Test recognition for a specific identity')
    test_parser.add_argument('--image', type=str,
                           help='Image for testing')
    test_parser.add_argument('--name', type=str,
                           help='Name of the identity to test (optional)')
    test_parser.add_argument('--detection-threshold', type=float,
                           help='Detection confidence threshold')
    test_parser.add_argument('--recognition-threshold', type=float,
                           help='Recognition similarity threshold')
    test_parser.add_argument('--no-display', action='store_true',
                           help='Do not display the result')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'list':
            list_identities(args)
        elif args.command == 'add':
            add_identity(args)
        elif args.command == 'remove':
            remove_identity(args)
        elif args.command == 'update':
            update_identity(args)
        elif args.command == 'rename':
            rename_identity(args)
        elif args.command == 'export':
            export_database(args)
        elif args.command == 'import':
            import_database(args)
        elif args.command == 'backup':
            backup_database(args)
        elif args.command == 'test':
            test_identity(args)
    except Exception as e:
        logger.exception(f"Error executing command: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()