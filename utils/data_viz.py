import zipfile
import os
from random import sample
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to read landmarks from a .pts file
def read_landmarks(pts_file):
    landmarks = []
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = map(float, line.strip().split())
            landmarks.append((x, y))
    return np.array(landmarks)

# Function to draw landmarks on an image
def draw_landmarks(image, landmarks,c=(0, 255, 0),size=None):
    for (x, y) in landmarks:
        if size==None:
            cv2.circle(image, (int(x), int(y)), int(image.shape[1]/150), c, -1)  # Draw a green circle at each landmark
        else:
            cv2.circle(image, (int(x), int(y)), size, c, -1)  # Draw a green circle at each landmark
            
        
        
# Compute the parameters of the bounding box of facial landmarks
def compute_bounding_box(landmarks):
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, width, height

def draw_bounding_box(image, landmarks):
    # Compute bounding box parameters
    min_x, min_y, width, height = compute_bounding_box(landmarks)

    # Draw bounding box
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(min_x + width), int(min_y + height)), (255, 0, 0),int(image.shape[1]/150))
    
    
    
# Function to widen the bounding box by a percentage
def widen_bounding_box(min_x, min_y, width, height, percentage=30):
    new_width = width + width * (percentage / 100.0)
    new_height = height + height * (percentage / 100.0)
    new_min_x = min_x - (new_width - width) / 2
    new_min_y = min_y - (new_height - height) / 2
    return new_min_x, new_min_y, new_width, new_height

# Function to crop and resize the image
def crop_and_resize(image, min_x, min_y, width, height, target_size=(128, 128)):
    # Ensure the bounding box stays within the image bounds
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image.shape[1], min_x + width)
    max_y = min(image.shape[0], min_y + height)

    # Check if the bounding box is valid
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid bounding box, no overlap with the image.")

    crop_img = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    resized_img = cv2.resize(crop_img, target_size)
    return resized_img