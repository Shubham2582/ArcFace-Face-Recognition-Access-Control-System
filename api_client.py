import os
import sys
import json
import base64
import argparse
import requests
import cv2
import numpy as np
from pprint import pprint


def encode_image_to_base64(image_path):
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_file_request(url, image_path, threshold=None):
    """
    Send image file to recognition API.

    Args:
        url: API endpoint URL
        image_path: Path to image file
        threshold: Recognition threshold (optional)

    Returns:
        API response
    """
    with open(image_path, "rb") as f:
        files = {"image": f}
        params = {}
        if threshold is not None:
            params["threshold"] = threshold

        response = requests.post(url, files=files, params=params)
        return response.json()


def send_base64_request(url, image_path, threshold=None):
    """
    Send base64 encoded image to recognition API.

    Args:
        url: API endpoint URL
        image_path: Path to image file
        threshold: Recognition threshold (optional)

    Returns:
        API response
    """
    # Encode image to base64
    encoded_image = encode_image_to_base64(image_path)

    # Prepare request
    data = {"image_base64": encoded_image}
    params = {}
    if threshold is not None:
        params["threshold"] = threshold

    # Send request
    response = requests.post(url, json=data, params=params)
    return response.json()


def display_results(result, image_path):
    """
    Display recognition results with image.

    Args:
        result: Recognition result from API
        image_path: Path to original image
    """
    if not result["success"]:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Create a copy for annotation
    annotated = img.copy()

    # Get face bbox if available
    if "face_bbox" in result:
        bbox = result["face_bbox"]
        x1, y1, x2, y2 = bbox

        # Determine color based on recognition
        color = (
            (0, 255, 0) if result["recognized"] else (0, 0, 255)
        )  # Green if recognized, red otherwise

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Get person name and confidence
        person_name = result["person_name"]
        confidence = result["confidence"]

        # Draw text
        text = f"{person_name}: {confidence:.2f}"
        cv2.putText(
            annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        # Draw top matches
        if "top_matches" in result:
            y_pos = 40
            for i, match in enumerate(result["top_matches"][:3]):
                match_text = f"{i+1}. {match['name']}: {match['score']:.2f}"
                cv2.putText(
                    annotated,
                    match_text,
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
                y_pos += 30

    # Display image
    window_name = "Recognition Result"
    cv2.imshow(window_name, annotated)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face Recognition API Client")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--url", default="http://localhost:5000/api/recognize", help="API endpoint URL"
    )
    parser.add_argument(
        "--method", choices=["file", "base64"], default="file", help="Upload method"
    )
    parser.add_argument("--threshold", type=float, help="Recognition threshold")
    parser.add_argument(
        "--display", action="store_true", help="Display results with image"
    )

    args = parser.parse_args()

    # Verify image path exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1

    print(f"Sending {args.method} request to {args.url}...")

    # Send request based on method
    if args.method == "file":
        result = send_file_request(args.url, args.image, args.threshold)
    else:
        result = send_base64_request(args.url, args.image, args.threshold)

    # Print results
    print("\nAPI Response:")
    pprint(result)

    # Display results if requested
    if args.display and result["success"]:
        display_results(result, args.image)

    return 0


if __name__ == "__main__":
    sys.exit(main())
