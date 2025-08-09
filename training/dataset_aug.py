'''Script to augment an input dataset by changing the brightness and contrast of images.'''
import cv2 as cv
import os
import argparse
import numpy as np
import random

def augment_dataset(input_path, output_path, num_img_per_image=5):
    '''Function to augment images in input_path by changing brightness and contrast, and save the entire dataset with the new images in output_path'''
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Iterate through images in input_path folder
    for img in os.listdir(input_path):
        img_path = os.path.join(input_path, img)
        if os.path.isfile(img_path):
            print("Processing:", img)
            # Load image
            image = cv.imread(img_path)
            if image is None:
                print(f"ERROR: Failed to open image file {img_path}")
                continue
            # Augment contrast and brightness randomly to generate num_img_per_image new images
            # alpha > 1 increases contrast, alpha < 1 decreases contrast
            # beta > 0 increases brightness, beta < 0 decreases brightness
            for i in range(num_img_per_image):
                alpha = random.uniform(0.5, 1.5)
                beta = random.randint(-50, 50)
                new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
                # Save augmented image
                new_img_path = os.path.join(output_path, f"{os.path.splitext(img)[0]}_aug_{i}.png")
                img_save = cv.imwrite(new_img_path, new_image)
                if not img_save:
                    print(f"ERROR: Failed to save augmented image {new_img_path}")
    print("Dataset augmentation completed.")
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset folder containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the augmented dataset")
    parser.add_argument("--num_img_per_image", type=int, default=5, help="Number of augmented images per original image")
    args = parser.parse_args()
    
    augment_dataset(args.input_path, args.output_path, args.num_img_per_image)