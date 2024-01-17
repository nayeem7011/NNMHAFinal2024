import cv2
import numpy as np
import math
import random
import csv
import os

# Define the file to save images
image_folder = 'D:\\TU Dresden\\Semester 5\\NNMHA\\generated_images'
csv_filename = 'D:\\TU Dresden\\Semester 5\\NNMHA\\rectangle_angles.csv'

# Create the image folder if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

angles = []

for i in range(1200):
    # Define the properties of the image and rectangle
    width, height = 600, 400
    rectangle_width, rectangle_height = int(0.6 * width), int(0.6 * height)

    # Generate a random floating-point number for the angle
    angle_degrees = random.uniform(-10, 10)

    # Calculate the radius of the bounding circle of the rectangle
    bounding_radius = math.sqrt((rectangle_width / 2)**2 + (rectangle_height / 2)**2)

    # Determine the safe area for the center to ensure the rectangle stays within the image
    safe_x_margin = bounding_radius
    safe_y_margin = bounding_radius

    min_x = safe_x_margin
    max_x = width - safe_x_margin
    min_y = safe_y_margin
    max_y = height - safe_y_margin

    # Generate a random center within the safe bounds
    center_x = random.uniform(min_x, max_x)
    center_y = random.uniform(min_y, max_y)

    # Create a blank grayscale image with a mid-gray background
    background_gray_level = 128  # Mid-gray background
    image = np.full((height, width), background_gray_level, dtype=np.uint8)

    # Generate a random grayscale level for the rectangle
    rectangle_gray_level = random.randint(0, 255)

    # Calculate the corner points of the rectangle based on the new random center
    rect_points = np.array([
        [center_x - rectangle_width / 2, center_y - rectangle_height / 2],
        [center_x + rectangle_width / 2, center_y - rectangle_height / 2],
        [center_x + rectangle_width / 2, center_y + rectangle_height / 2],
        [center_x - rectangle_width / 2, center_y + rectangle_height / 2]
    ], dtype=np.float32)

    # Create and apply a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
    rotated_points = cv2.transform(rect_points.reshape(-1, 1, 2), rotation_matrix)

    # Draw the rotated rectangle in a random grayscale color
    cv2.fillPoly(image, [np.int32(rotated_points)], color=(rectangle_gray_level))

    # Save the image and angle
    angles.append((f'rotated_rectangle_{i}', angle_degrees))
    image_filename = os.path.join(image_folder, f'rotated_rectangle_{i}.png')
    cv2.imwrite(image_filename, image)

# Save the angles to a CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filename', 'angle'])
    csv_writer.writerows(angles)

print(f'Angles saved to {csv_filename}')
print(f'Images saved to {image_folder}')
