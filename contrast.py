import cv2
import os

# Input and output directories
input_folder = 'prepro/stain_chosen'
output_folder = 'prepro/stain_contrast'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to enhance contrast
def enhance_contrast(img, clip_limit=2.0, grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)

    # Merge the enhanced L channel back with the A and B channels
    enhanced_lab = cv2.merge((l_enhanced, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    # Build the full path to the image
    img_path = os.path.join(input_folder, filename)

    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Read the color image
        img = cv2.imread(img_path)

        # Step 1: Enhance contrast
        contrast_img = enhance_contrast(img, clip_limit=8.0, grid_size=(4, 4))

        # Step 2: Convert to grayscale
        grayscale_img = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

        # Save the enhanced grayscale image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, grayscale_img)

        print(f"Processed and saved: {output_path}")

print("Processing complete.")