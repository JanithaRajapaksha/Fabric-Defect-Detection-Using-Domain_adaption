import cv2
import os
import numpy as np
import random

# Input and output directories
input_folder = 'source/train/stain'
output_folder = 'target/train/stain'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to add a random color overlay
def add_random_color_overlay(gray_img):
    # Convert grayscale to BGR
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Generate random BGR color
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Create a solid color image
    color_layer = np.full_like(bgr_img, random_color, dtype=np.uint8)

    # Blend the grayscale and color layer
    blended_img = cv2.addWeighted(bgr_img, 0.5, color_layer, 0.5, 0)
    return blended_img

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    # Build the full path to the image
    img_path = os.path.join(input_folder, filename)

    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Read the image in grayscale
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Add random color overlay
        colored_img = add_random_color_overlay(gray_img)

        # Save the colored image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, colored_img)

        print(f"Processed and saved: {output_path}")

print("Processing complete.")