import logging
from gemini_kit import detect_2d, segmentation, pointing, check_api_key

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 1. 2D Object Detection (Bounding Boxes)
# ==========================================

def example_detection():
    """
    Detects objects (bounding boxes).
    Models supported: All.
    """
    print("\n--- Running Example 1. 2D Object Detection ---")

    # Image: A kitchen
    image_url = "https://images.unsplash.com/photo-1588854337221-4cf9fa96059c?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

    json_result, visual_image = detect_2d(
        model="gemini-2.5-pro", 
        prompt="Detect all metal objects in the image",
        image_path=image_url,
        visual=True,
        output_path="output_samples",
        max_retries=5
    )

    print(json_result)  # Print the raw JSON results
    if visual_image:
        visual_image.show()  # Display the visual image


# ==========================================
# 2. Instance Segmentation (Masks)
# ==========================================

def example_segmentation():
    """
    Segments objects (pixel-level masks).
    Models supported: Gemini 2.5 family only.
    """
    print("\n--- Running Example 2: Segmentation ---")

    # Image: A parking lot
    image_url = "https://images.unsplash.com/photo-1593280405106-e438ebe93f5b?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

    json_result, visual_image = segmentation(
        model="gemini-2.5-pro", 
        prompt="Segment all red cars in the image",
        image_path=image_url,
        visual=True,
        output_path="output_samples",
        max_retries=5
    )

    print(json_result)  # Print the JSON results
    if visual_image:
        visual_image.show()  # Display the visual image


# ==========================================
# 3. Pointing (Basic & Spatial)
# ==========================================

def example_pointing_basic():
    """
    Points to elements in the image.
    Models supported: All.
    """
    print("\n--- Running Example 3. Pointing ---")

    # Image: A drawing of a bedroom
    image_url = "https://images.unsplash.com/photo-1763669029223-74f911a9e08b?q=80&w=1107&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

    json_result, visual_image = pointing(
        model="gemini-3-pro-preview", 
        prompt="Point at all furniture in the image. Label where they are located.",
        image_path=image_url,
        visual=True,
        output_path="output_samples",
        max_retries=5
    )

    print(json_result)  # Print the JSON results
    if visual_image:
        visual_image.show()  # Display the visual image

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    if check_api_key():
        # Run examples
        # example_detection()
        # example_segmentation()
        # example_pointing_basic()
        
        print("\nAll spatial examples finished.")