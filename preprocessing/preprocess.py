import cv2 as cv
import os
import argparse

def preprocess(input_path, output_path):
    '''Function to extract frames from mp4 experiment files in input_path and save them in output_path'''
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Iterate through experiment folders in path
    for folder in os.listdir(input_path):
        # Check if folder is a directory
        if os.path.isdir(os.path.join(input_path, folder)):
            # Check for mp4 file in the folder and extract frames
            for file in os.listdir(os.path.join(input_path, folder)):
                if file.endswith(".mp4"):
                    print("Processing:", folder)
                    # Load video file
                    video_path = os.path.join(input_path, folder, file)
                    video_capture = cv.VideoCapture(video_path)
                    # Check if video is empty
                    if not video_capture.isOpened() or os.path.getsize(video_path) == 0:
                        print(f"Error: Failed to open video file {video_path}")
                        continue
                    # Create folder for frames if it doesn't exist
                    if not os.path.exists(os.path.join(output_path, folder)):
                        os.makedirs(os.path.join(output_path, folder))
                    # Extract frames from video
                    count = 0
                    while (video_capture.isOpened()):
                        ret, frame = video_capture.read()
                        if ret == False:
                            break
                        frame_path = os.path.join(output_path, folder, f"{count:04d}.png")
                        count += 1
                        img_save = cv.imwrite(frame_path, frame)
                        if not img_save:
                            print(f"Error: Failed to save frame {count:04d}")
                    video_capture.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="demonstration")
    parser.add_argument("--output_path", type=str, default="preprocessed_data")
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)