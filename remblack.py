import cv2
import numpy as np

def replace_dark_with_white(image, threshold=30):
    """
    Replace all colors close to black with white.
    
    Args:
        image (np.array): The input image.
        threshold (int): The maximum intensity value to consider a color close to black.
                         Any pixel with all RGB values below this threshold will be considered.
    
    Returns:
        np.array: The modified image with dark colors replaced by white.
    """
    # Create a mask for colors close to black (all RGB values below the threshold)
    mask = np.all(image < threshold, axis=-1)

    # Replace the selected dark pixels with white (255, 255, 255)
    image[mask] = [255, 255, 255]

    return image

# Example usage
image = cv2.imread('second.png')  # Load your image

# Replace colors close to black with white, using a threshold of 30 (you can adjust this)
image_white_bg = replace_dark_with_white(image, threshold=30)

# Save the modified image
cv2.imwrite('image_white_background.png', image_white_bg)
print("Final output saved as image_white_background.png in present working directory")
