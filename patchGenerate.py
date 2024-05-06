import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()

import cv2
import numpy as np
import random

trainPath = "../train_images/"

def block_generate(input_channel):
    # Block size
    block_size = 250

    # Calculate padding needed
    pad_height = (block_size - input_channel.shape[0] % block_size) % block_size
    pad_width = (block_size - input_channel.shape[1] % block_size) % block_size

    # Apply padding
    padded_matrix = np.pad(input_channel, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Now the padded matrix should be divisible by 250 both ways
    num_blocks_vert = padded_matrix.shape[0] // block_size
    num_blocks_horz = padded_matrix.shape[1] // block_size

    # List to hold the blocks
    blocks = []

    # Split the padded matrix into blocks
    for i in range(num_blocks_vert):
        row_start = i * block_size
        row_end = row_start + block_size
        for j in range(num_blocks_horz):
            col_start = j * block_size
            col_end = col_start + block_size
            block = padded_matrix[row_start:row_end, col_start:col_end]
            total_elements = block.size
            non_zero_count = np.count_nonzero(block)
            if non_zero_count >= (total_elements/2):
                blocks.append(block)
    return blocks


for filename in os.listdir(trainPath):
    # Load the image 
    image_path = os.path.join(trainPath, filename)
    imageId, extension = os.path.splitext(filename)
    image = cv2.imread(image_path) 

    base_directory = '../preprocess_images_threshold'
    sample_directory = os.path.join(base_directory, f"sample_{imageId}")
    os.makedirs(sample_directory, exist_ok=True)

    blocks0 = block_generate(image[:,:,0])
    blocks1 = block_generate(image[:,:,1])
    blocks2 = block_generate(image[:,:,2])

    indices = list(range(len(blocks0)))
    sampled_indices = random.sample(indices, 100)
    for i in range(0, len(sampled_indices)):
        r = blocks0[sampled_indices[i]]
        g = blocks1[sampled_indices[i]]
        b = blocks2[sampled_indices[i]]
        rgb_array = np.stack([r, g, b], axis=-1)
        patch_filename = f"{imageId}_{i}.png"
        patch_path = os.path.join(sample_directory, patch_filename)
        cv2.imwrite(patch_path, rgb_array)

