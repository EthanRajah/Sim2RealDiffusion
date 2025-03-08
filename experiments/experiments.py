import bagpy
from bagpy import bagreader
import pandas as pd
import os
import cv2
import numpy as np

def extract_bag_data(bag_file, output_dir):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read bag file
    b = bagreader(bag_file)

    # List available topics
    print("Available topics:", b.topic_table)

    # Extract image topic (replace with actual image topic)
    image_topic = "/rgb_publisher/color/image"
    image_data = b.message_by_topic(image_topic)
    # Save to png files
    for i, data in enumerate(image_data):
        image = data.data
        image = image.reshape((data.height, data.width, 3))
        image = image[:, :, ::-1]  # Convert BGR to RGB
        image = image.astype("uint8")
        image_path = os.path.join(output_dir, f"image_{i}.png")
        cv2.imwrite(image_path, image)

    print(f"Data saved to {image_data}")

def extract_images_from_csv(csv_file, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract and decode images
    for i, row in data.iterrows():
        try:
            # Convert binary data to a numpy array
            image_data = np.frombuffer(bytes.fromhex(row['data']), dtype=np.uint8)
            
            # Decode the image (assuming bgr8 encoding)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            # Save the image as PNG
            output_path = os.path.join(output_dir, f"image_{i:04d}.png")
            cv2.imwrite(output_path, image)
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error processing row {i}: {e}")

if __name__ == "__main__":
    bag_file = "rope_data.bag"  # Path to your bag file
    output_dir = "./extracted_data"
    #extract_bag_data(bag_file, output_dir)
    extract_images_from_csv("rope_data/rgb_publisher-color-image.csv", output_dir)
