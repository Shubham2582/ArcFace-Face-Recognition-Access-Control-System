"""
Image processing utilities.
"""

import logging
import cv2
import numpy as np
import os

logger = logging.getLogger(__name__)

def read_image(img_path):
    """
    Read image from path.
    
    Args:
        img_path: Path to image
    
    Returns:
        Image as numpy array or None if failed
    """
    if not os.path.exists(img_path):
        logger.warning(f"Image path does not exist: {img_path}")
        return None
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error reading image {img_path}: {e}")
        return None

def resize_image(img, width=None, height=None):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        img: Input image
        width: Target width
        height: Target height
    
    Returns:
        Resized image
    """
    if img is None:
        return None
    
    if width is None and height is None:
        return img
    
    h, w = img.shape[:2]
    
    if width is None:
        aspect_ratio = height / float(h)
        dim = (int(w * aspect_ratio), height)
    elif height is None:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))
    else:
        dim = (width, height)
    
    try:
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return img

def is_image_file(file_path):
    """
    Check if file is an image.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file is an image, False otherwise
    """
    if not os.path.isfile(file_path):
        return False
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    return file_ext in image_extensions

def draw_text_with_background(img, text, position, font_scale=0.7, 
                              thickness=2, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), padding=5):
    """
    Draw text with background on image.
    
    Args:
        img: Input image
        text: Text to draw
        position: Position (x, y) to draw text
        font_scale: Font scale
        thickness: Line thickness
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        padding: Padding around text
    
    Returns:
        Image with text
    """
    if img is None:
        return None
    
    # Make a copy of the image
    result = img.copy()
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    
    # Calculate background rectangle dimensions
    rect_x = position[0]
    rect_y = position[1] - text_height - padding
    rect_width = text_width + 2 * padding
    rect_height = text_height + 2 * padding + baseline
    
    # Draw background rectangle
    cv2.rectangle(result, 
                 (rect_x, rect_y), 
                 (rect_x + rect_width, rect_y + rect_height), 
                 bg_color, -1)
    
    # Draw text
    cv2.putText(result, text, 
               (position[0] + padding, position[1] - padding), 
               font, font_scale, text_color, thickness)
    
    return result

def draw_access_status(img, granted=False, position=(20, 50), font_scale=1.0, thickness=2):
    """
    Draw access status on image.
    
    Args:
        img: Input image
        granted: Whether access is granted
        position: Position (x, y) to draw text
        font_scale: Font scale
        thickness: Line thickness
    
    Returns:
        Image with access status
    """
    if img is None:
        return None
    
    # Make a copy of the image
    result = img.copy()
    
    if granted:
        text = "ACCESS GRANTED"
        color = (0, 255, 0)  # Green
    else:
        text = "ACCESS DENIED"
        color = (0, 0, 255)  # Red
    
    # Draw text with background
    result = draw_text_with_background(
        result, text, position, font_scale, thickness, 
        text_color=(255, 255, 255), bg_color=color)
    
    return result

def create_montage(images, num_cols=3, padding=10):
    """
    Create a montage from multiple images.
    
    Args:
        images: List of images
        num_cols: Number of columns
        padding: Padding between images
    
    Returns:
        Montage image
    """
    if not images:
        return None
    
    # Determine grid dimensions
    n_images = len(images)
    n_cols = min(num_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Ensure all images are the same size
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    
    # Get average height and width
    avg_height = sum(heights) // len(heights)
    avg_width = sum(widths) // len(widths)
    
    # Resize all images to average size
    resized_images = [resize_image(img, avg_width, avg_height) for img in images]
    
    # Create empty montage
    montage_height = n_rows * (avg_height + padding) + padding
    montage_width = n_cols * (avg_width + padding) + padding
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
    
    # Fill montage
    for i, img in enumerate(resized_images):
        row = i // n_cols
        col = i % n_cols
        
        y_start = row * (avg_height + padding) + padding
        y_end = y_start + avg_height
        
        x_start = col * (avg_width + padding) + padding
        x_end = x_start + avg_width
        
        montage[y_start:y_end, x_start:x_end, :] = img
    
    return montage