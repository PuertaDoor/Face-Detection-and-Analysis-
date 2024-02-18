import zipfile
import os
from random import sample
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .data_viz import*


def crop_resize_save(l_images,lm_to_show,idxs_show,path="/content/train_images",plot=False):
    fig=None
    axs=None
    if plot:
        fig, axs = plt.subplots(4, 3, figsize=(10, 10), layout='constrained')
    cpt_show=0
    for i in range(l_images.size):

        image = cv2.imread(l_images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = read_landmarks(lm_to_show[i])

        # Compute bounding box parameters
        min_x, min_y, width, height = compute_bounding_box(landmarks)

        # Widen the bounding box by 30%
        new_min_x, new_min_y, new_width, new_height = widen_bounding_box(min_x, min_y, width, height, percentage=30)

        # Crop and resize the image
        resized_image = crop_and_resize(image, new_min_x, new_min_y, new_width, new_height)
        if plot:
            if i in idxs_show:
                axs[int(cpt_show%4)][int(cpt_show%3)].imshow(resized_image)
                cpt_show+=1
        #axs[int(i%4)][int(i%3)].axis("off")

        # Save the resized image to disk
        cv2.imwrite(f"{path}/resized_image_{i}.jpg", resized_image)
        
        
# Function to scale landmark coordinates based on resizing factor
def scale_landmarks(landmarks, min_x, min_y, width, height, target_size=(128, 128)):
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height
    scaled_landmarks = ((landmarks - np.array([min_x, min_y])) * np.array([scale_x, scale_y]))
    return scaled_landmarks


def update_lm(im_to_show,lm_to_show,idxs_show,path="train_images",plot=False):
    fig=None
    axs=None
    if plot:
        fig, axs = plt.subplots(4, 3, figsize=(10, 10), layout='constrained')
    cpt_show=0
    for i in range(im_to_show.size):
        image = cv2.imread(im_to_show[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = read_landmarks(lm_to_show[i])

        # Compute bounding box parameters
        min_x, min_y, width, height = compute_bounding_box(landmarks)

        # Widen the bounding box by 30%
        new_min_x, new_min_y, new_width, new_height = widen_bounding_box(min_x, min_y, width, height, percentage=30)

        # Crop and resize the image
        resized_image = crop_and_resize(image, new_min_x, new_min_y, new_width, new_height)

        # Scale the landmarks to the resized image
        scaled_landmarks = scale_landmarks(landmarks, new_min_x, new_min_y, new_width, new_height, target_size=resized_image.shape[:2]).tolist()

        new_pts = [f"{lm[0]} {lm[1]}" for lm in scaled_landmarks]

        # Display the resized image with landmarks
        for (x, y) in scaled_landmarks:
            cv2.circle(resized_image, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        if plot:
            if i in idxs_show:
                axs[int(cpt_show%4)][int(cpt_show%3)].imshow(resized_image)
                axs[int(cpt_show%4)][int(cpt_show%3)].axis("off")
                cpt_show+=1

        
        
       
   
        
        with open(f"./content/"+path+"/resized_image_"+str(i)+".pts", "w") as f:
            f.write("\n".join(new_pts))
            
# Function to create cv2.KeyPoint objects for each landmark
def create_keypoints(mean_shape, patch_size=20):
    keypoints = []

    for (x, y) in mean_shape:
        # Create a cv2.KeyPoint object for each landmark
        keypoint = cv2.KeyPoint(x, y, patch_size)
        keypoints.append(keypoint)

    return keypoints
            
# Function to compute the mean shape of the face
def compute_mean_shape(data_folder,plot=False):
    # List all training images
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

    # Initialize variables to accumulate landmark coordinates
    sum_landmarks = np.zeros((68, 2), dtype=float)
    total_images = len(image_files)
    L_landmarks=[]
    # Accumulate landmark coordinates
    for image_file in image_files:
        image_path = os.path.join(data_folder, image_file)
        landmarks_path = os.path.join(data_folder, image_file.replace('.jpg', '.pts'))

        landmarks = read_landmarks(landmarks_path)
        L_landmarks.append(landmarks)
        sum_landmarks += landmarks

    # Compute the mean shape
    mean_landmarks = sum_landmarks / total_images
    
    keypoints = create_keypoints(mean_landmarks.copy())

    # Visualize the keypoints on an image
    image_with_keypoints = np.ones((128, 128, 3), dtype=np.uint8) * 255  # Create a white image for visualization

    for keypoint in keypoints:
        x, y = map(int, keypoint.pt)
        cv2.circle(image_with_keypoints, (x, y), 1, (255, 0, 0), -1)  # Draw a red circle at each keypoint
    
    if plot:
        plt.imshow(image_with_keypoints)
        plt.title("Mean shape of the face")

    return mean_landmarks,L_landmarks

# Function to compute the mean shape of the face
def draw_mean_on_faces(data_folder,mean_lm,idxs_show):
    # List all training images
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
    fig, axs = plt.subplots(4, 3, figsize=(10, 10), layout='constrained')

    cpt_show=0
    # Accumulate landmark coordinates
    for i,image_file in enumerate(image_files):
        image_path = os.path.join(data_folder, image_file)
        
        image = cv2.imread(image_path)
        
        landmarks_path = os.path.join(data_folder, image_file.replace('.jpg', '.pts'))

        landmarks = read_landmarks(landmarks_path)

    
        draw_landmarks(image, landmarks,size=1)
        draw_landmarks(image, mean_lm,c=(255,0,0),size=1)
    
        if i in idxs_show:
            axs[int(cpt_show%4)][int(cpt_show%3)].imshow(image)
            cpt_show+=1
        
        
        
# Function to initialize the mean shape in each image and generate perturbations
def initialize_and_perturbate(image_path, mean_shape, num_perturbations=10, scale_amplitude=0.2, translation_amplitude=20):
    # Read image and landmarks
    image = cv2.imread(image_path)
    landmarks_path = image_path.replace('.jpg', '.pts')
    landmarks = read_landmarks(landmarks_path)

    # Initialize the mean shape in the image
    mean_landmarks = mean_shape.copy()

    # Generate perturbations
    perturbations = []
    for _ in range(num_perturbations):
        # Generate random scaling factor in the range [1 - scale_amplitude, 1 + scale_amplitude]
        scaling_factor = np.random.uniform(1 - scale_amplitude, 1 + scale_amplitude)

        # Generate random translation in the range [-translation_amplitude, translation_amplitude]
        translation_x = np.random.uniform(-translation_amplitude, translation_amplitude)
        translation_y = np.random.uniform(-translation_amplitude, translation_amplitude)

        # Apply scaling and translation to mean shape
        perturbed_landmarks = (mean_landmarks * scaling_factor) + np.array([translation_x, translation_y])

        perturbations.append(perturbed_landmarks)

    return image, landmarks, perturbations


def show_perturbations(image, landmarks, perturbations):
    fig, axs = plt.subplots(3, 5, figsize=(10, 6), layout='constrained')
    
    fond_blanc=np.uint(np.ones((128,128))*255)
    axs[0][0].imshow(fond_blanc,vmin=0,vmax=255,cmap='gray')
    axs[0][0].axis("off")
    axs[0][2].imshow(fond_blanc,vmin=0,vmax=255,cmap='gray')
    axs[0][2].axis("off")
    axs[0][4].imshow(fond_blanc,vmin=0,vmax=255,cmap='gray')
    axs[0][4].axis("off")
    
    axs[0][1].imshow(image)
    axs[0][1].axis("off")
    axs[0][1].set_title("original")
    
    image_lm=image.copy()
    draw_landmarks(image_lm, landmarks,size=1)
    axs[0][3].imshow(image_lm)
    axs[0][3].axis("off")
    axs[0][3].set_title("original landmarks")
    
    for i,perturb in enumerate(perturbations):
        image_tmp=image.copy()
        draw_landmarks(image_tmp,perturb,c=(255,0,0),size=2)
        axs[(i//5)+1][i%5].imshow(image_tmp)
        axs[(i//5)+1][i%5].axis("off")
        axs[(i//5)+1][i%5].set_title("perturbation: "+str(i))
        
        

def crop_resize_save_enhanced(l_images,lm_to_show,idxs_show,path="/content/train_images",plot=False):
    #(image, min_x, min_y, width, height, target_size=(128, 128)):
    translation_amplitude=20
    scaling_factor=0.6
    cpt_img=0
    
    fig=None
    axs=None
    if plot:
        fig, axs = plt.subplots(5, 5, layout='constrained')
    cpt_show=0
    for i in range(l_images.size):
        for idx_transfo in range(5):
            translation_x = np.random.uniform(-translation_amplitude, translation_amplitude)
            translation_y = np.random.uniform(-translation_amplitude, translation_amplitude)
            scaling_factor = np.random.uniform(1, 1 + scaling_factor)
            new_width_percentage=np.uint(30*scaling_factor)

            image = cv2.imread(l_images[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = read_landmarks(lm_to_show[i])
            landmarks=np.array(landmarks)
            
            # Compute bounding box parameters
            min_x, min_y, width, height = compute_bounding_box(landmarks)

            # Widen the bounding box by 30%
            min_x+=translation_x
            min_y+=translation_y
            
            
            new_min_x, new_min_y, new_width, new_height = widen_bounding_box(min_x, min_y, width, height, percentage=new_width_percentage)
            
            # Crop and resize the image
            resized_image = crop_and_resize(image, new_min_x, new_min_y, new_width, new_height)
            
            scaled_landmarks = scale_landmarks(landmarks, new_min_x, new_min_y, new_width, new_height, target_size=resized_image.shape[:2]).tolist()
            
            if plot:
                if i<5:
                    resized_image_tmp=resized_image.copy()
                    draw_landmarks(resized_image_tmp, scaled_landmarks,size=2,c=(0, 255,0))
                    axs[i][idx_transfo].imshow(resized_image_tmp)
                    cpt_show+=1
            #axs[int(i%4)][int(i%3)].axis("off")

            # Save the resized image to disk
            cv2.imwrite("."+path+"/resized_image_"+str(cpt_img)+".jpg", resized_image)
            
            with open("."+path+"/resized_image_"+str(cpt_img)+".pts", "w") as f:
                new_pts = [f"{lm[0]} {lm[1]}" for lm in scaled_landmarks]
                f.write("\n".join(new_pts))
            cpt_img+=1
            
# Function to initialize the mean shape in each image and generate perturbations
def perturbate_meanshape(mean_shape, scale_amplitude=0.2, translation_amplitude=20):

    # Initialize the mean shape in the image
    mean_landmarks = mean_shape.copy()

    scaling_factor = np.random.uniform(1 - scale_amplitude, 1 + scale_amplitude)

    # Generate random translation in the range [-translation_amplitude, translation_amplitude]
    translation_x = np.random.uniform(-translation_amplitude, translation_amplitude)
    translation_y = np.random.uniform(-translation_amplitude, translation_amplitude)

    # Apply scaling and translation to mean shape
    perturbed_landmarks = (mean_landmarks * scaling_factor) + np.array([translation_x, translation_y])
    
    return perturbed_landmarks

