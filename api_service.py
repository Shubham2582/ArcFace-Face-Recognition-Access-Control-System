#!/usr/bin/env python3
"""
API service for the face recognition system.
This API allows for face recognition by accepting image uploads.
"""

import os
import sys
import logging
import json
import base64
import time
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import ssl

# Configure logging
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f'api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Import from our modules
from config import settings

from src.face.detector import FaceDetector
from src.face.embedder import FaceEmbedder
from src.database.embeddings_db import EmbeddingsDatabase
from src.database.faiss_db import FaissDatabase

# Initialize Flask application
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Global variables for lazy loading
detector = None
embedder = None
database = None

def initialize_components():
    """Initialize components only when needed (lazy loading)."""
    global detector, embedder, database
    
    if detector is None:
        logger.info("Initializing face detector...")
        detector = FaceDetector(detection_threshold=settings.DETECTION_THRESHOLD)
        
        # Verify detector has required methods
        required_methods = ["detect_faces", "get_largest_face", "draw_face_locations"]
        missing_methods = []
        for method in required_methods:
            if not hasattr(detector, method):
                missing_methods.append(method)
                logger.error(f"Detector missing required method: {method}")

        if missing_methods:
            logger.error(f"Detector is missing critical methods: {missing_methods}")
            raise Exception("Detector initialization failed")
            
    if embedder is None:
        logger.info("Initializing face embedder...")
        embedder = FaceEmbedder()

    if database is None:
        logger.info("Initializing database...")
        # Initialize database (choose between FAISS and standard)
        if settings.USE_FAISS:
            database = FaissDatabase()
        else:
            database = EmbeddingsDatabase()

        # Check database at startup
        db_info = database.get_database_info()
        if db_info["num_identities"] == 0:
            if settings.IS_PRODUCTION:
                logger.error("Database is empty in production environment!")
                raise Exception("Database is empty. Please ensure face_db.pkl contains valid data.")
            else:
                logger.warning("Database is empty. Please add identities first.")
                raise Exception("Database is empty. Please ensure face_db.pkl contains valid data.")
        else:
            logger.info(f"Database loaded with {db_info['num_identities']} identities")

# Initialize only the database at startup to check if it exists
try:
    if settings.USE_FAISS:
        database = FaissDatabase()
    else:
        database = EmbeddingsDatabase()
    
    db_info = database.get_database_info()
    if db_info["num_identities"] == 0:
        logger.warning("Database is empty. Face recognition will not work until models are loaded.")
    else:
        logger.info(f"Database ready with {db_info['num_identities']} identities")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    database = None


def process_image_recognition(image_data, recognition_threshold=None):
    """
    Process image and perform face recognition.

    Args:
        image_data: Image data as numpy array
        recognition_threshold: Optional recognition threshold

    Returns:
        Dictionary with recognition results
    """
    if recognition_threshold is None:
        recognition_threshold = settings.RECOGNITION_THRESHOLD

    try:
        # Initialize components only when needed (lazy loading)
        initialize_components()
        
        # Start timing
        start_time = time.time()

        # Add debug logging
        logger.debug("Attempting to detect face...")

        try:
            face = detector.get_largest_face(image_data)
        except Exception as face_error:
            logger.error(f"Face detection error: {face_error}")
            face = None

        if face is None:
            logger.info("No face detected in image")
            return {
                "success": False,
                "error": "No face detected in image",
                "processing_time": time.time() - start_time,
            }

        # Get embedding
        embedding = embedder.get_embedding(face)
        if embedding is None:
            return {
                "success": False,
                "error": "Failed to get face embedding",
                "processing_time": time.time() - start_time,
            }

        # Match against database
        name, score = database.find_match(embedding, recognition_threshold)

        # Get all similarity scores
        all_scores = database.get_all_similarity_scores(embedding)
        top_matches = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate bbox coordinates for response
        bbox = face.bbox.astype(int).tolist()

        # Prepare response
        result = {
            "success": True,
            "recognized": name is not None,
            "person_name": name if name else "Unknown",
            "confidence": float(score),
            "face_bbox": bbox,
            "top_matches": [{"name": p, "score": float(s)} for p, s in top_matches],
            "processing_time": time.time() - start_time,
        }

        return result

    except Exception as e:
        logger.exception(f"Error in image processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time": (
                time.time() - start_time if "start_time" in locals() else 0
            ),
        }


@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    """API endpoint for face recognition."""
    # Check if image file is present in request
    if "image" not in request.files and "image_base64" not in request.json:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No image provided. Send an image file or base64 encoded image.",
                }
            ),
            400,
        )

    # Get recognition threshold from request (optional)
    try:
        recognition_threshold = float(
            request.args.get("threshold", settings.RECOGNITION_THRESHOLD)
        )
    except ValueError:
        recognition_threshold = settings.RECOGNITION_THRESHOLD

    # Process file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"}), 400

        # Save uploaded file
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{int(time.time())}_{filename}"
            )
            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)
            if img is None:
                return (
                    jsonify({"success": False, "error": "Failed to read image file"}),
                    400,
                )

            logger.info(f"Processing uploaded file: {filename}")

            # Clean up file after processing to save space
            try:
                os.remove(filepath)
            except:
                pass  # Ignore errors in cleanup

        except Exception as e:
            logger.error(f"Error processing file upload: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Process base64 encoded image
    elif "image_base64" in request.json:
        try:
            # Decode base64 image
            image_data = request.json["image_base64"]
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return (
                    jsonify(
                        {"success": False, "error": "Failed to decode base64 image"}
                    ),
                    400,
                )

            logger.info("Processing base64 encoded image")
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Error processing base64 image: {str(e)}",
                    }
                ),
                400,
            )

    # Process the image
    result = process_image_recognition(img, recognition_threshold)

    # Return the result
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        if database is None:
            return jsonify({"status": "error", "error": "Database not initialized"}), 500
            
        db_info = database.get_database_info()
        return jsonify(
            {
                "status": "ok",
                "time": datetime.now().isoformat(),
                "database_size": len(db_info["identities"]),
                "version": "1.0.0",
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check system status."""
    try:
        return jsonify({
            "database_initialized": database is not None,
            "detector_initialized": detector is not None,
            "embedder_initialized": embedder is not None,
            "is_production": settings.IS_PRODUCTION,
            "db_path": settings.DB_PATH,
            "db_file_exists": os.path.exists(settings.DB_PATH),
            "ctx_id": settings.CTX_ID,
            "model_name": settings.MODEL_NAME
        })
    except Exception as e:
        logger.error(f"Debug info failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/database/info", methods=["GET"])
def database_info():
    """Get database information."""
    try:
        if database is None:
            return jsonify({"success": False, "error": "Database not initialized"}), 500
            
        db_info = database.get_database_info()
        return jsonify(
            {
                "success": True,
                "database_type": "FAISS" if settings.USE_FAISS else "Standard",
                "num_identities": db_info["num_identities"],
                "identities": db_info["identities"],
            }
        )
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Root endpoint for basic information."""
    return jsonify(
        {
            "service": "Face Recognition API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": [
                {"path": "/", "method": "GET", "description": "Service information"},
                {"path": "/api/health", "method": "GET", "description": "Health check"},
                {
                    "path": "/api/database/info",
                    "method": "GET",
                    "description": "Database information",
                },
                {
                    "path": "/api/recognize",
                    "method": "POST",
                    "description": "Face recognition endpoint",
                },
            ],
        }
    )


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))

    # Get host from environment or use default
    host = os.environ.get("HOST", "0.0.0.0")

    # Check for production mode
    production_mode = os.environ.get("PRODUCTION", "false").lower() == "true"

    logger.info(
        f"Starting API server on {host}:{port} in {'production' if production_mode else 'development'} mode"
    )

    # Log database status
    if database is not None:
        try:
            db_info = database.get_database_info()
            logger.info(f"Database ready with {db_info['num_identities']} identities")
        except Exception as e:
            logger.error(f"Database check failed: {e}")
    else:
        logger.warning("Database not initialized at startup")

    if production_mode:
        # In production mode, use Gunicorn instead (this won't execute)
        logger.info(
            "In production mode, use: gunicorn --workers=4 --bind=0.0.0.0:5000 api_service:app"
        )
    else:
        # Development mode
        app.run(host=host, port=port, debug=False)
