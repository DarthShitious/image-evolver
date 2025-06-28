import os
import re
import cv2


def pngs_to_mp4(input_dir, output_path, fps=30):
    """
    Creates an MP4 video from PNG images in a directory that have numbered suffixes.

    Args:
        input_dir (str): Path to directory containing PNG files.
        output_path (str): Path for the output MP4 file.
        fps (int, optional): Frames per second for the output video. Default is 30.
    """

    # Regex to find numeric suffix
    pattern = re.compile(r"(.*?)(\d+)\.png$", re.IGNORECASE)

    # Gather and sort files
    png_files = []
    for f in os.listdir(input_dir):
        match = pattern.match(f)
        if match:
            prefix, num = match.groups()
            png_files.append((int(num), os.path.join(input_dir, f)))

    if not png_files:
        raise ValueError("No PNG files with numeric suffix found in the directory.")

    png_files.sort()

    # Read first image to get frame size
    first_img = cv2.imread(png_files[0][1])
    if first_img is None:
        raise ValueError(f"Could not read image: {png_files[0][1]}")
    height, width, layers = first_img.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for _, file_path in png_files:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: skipping unreadable file {file_path}")
            continue
        out.write(img)

    out.release()
    print(f"Video saved to {output_path}")