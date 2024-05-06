import os
import numpy as np
import random
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.spatial import KDTree
Image.MAX_IMAGE_PIXELS = None  # This removes the limit entirely

# Define paths
base_path = "/scratch1/yuqiuwan/CSCI567/"
train_path = os.path.join(base_path, "train_images1/")
thumbnail_path = os.path.join(base_path, "quarantine_room/train_thumbnails/")
output_path = os.path.join(base_path, "mask_images/")
os.makedirs(output_path, exist_ok=True)

def compute_matter_mask(path_thumbnail):
    thumbnail_img = Image.open(path_thumbnail).convert('RGB')
    arr = np.array(thumbnail_img)
    hsv = rgb2hsv(arr)
    threshold_h = threshold_otsu(hsv[:, :, 0])
    threshold_s = threshold_otsu(hsv[:, :, 1])
    mask = (hsv[:, :, 0] > threshold_h) & (hsv[:, :, 1] > threshold_s)
    return mask.astype(np.uint8)

def get_tiles_coordinates(thumb_path, fullres_dimensions, tile_size=250):
    mask = compute_matter_mask(thumb_path)
    thumb_img = Image.open(thumb_path)
    thumb_w, thumb_h = thumb_img.size
    scale_x, scale_y = fullres_dimensions[0] / thumb_w, fullres_dimensions[1] / thumb_h
    coords = np.argwhere(mask).tolist()
    scaled_coords = [(int(x * scale_x), int(y * scale_y)) for x, y in coords]
    return [coord for coord in scaled_coords if coord[0] + tile_size <= fullres_dimensions[0] and coord[1] + tile_size <= fullres_dimensions[1]]

def filter_close_coordinates(coordinates, min_distance):
    tree = KDTree(coordinates)
    filtered_indices = set()

    for i, coord in enumerate(coordinates):
        if i in filtered_indices:
            continue  # Skip already filtered coordinates
        indices = tree.query_ball_point(coord, min_distance)
        filtered_indices.update(indices)
        filtered_indices.remove(i)  # Keep current point
    return [coordinates[i] for i in range(len(coordinates)) if i not in filtered_indices]


def is_valid_patch(patch, white_threshold=230, black_threshold=15):
    np_patch = np.array(patch)
    valid = np.logical_and(np_patch < white_threshold, np_patch > black_threshold).all(axis=-1)
    return np.mean(valid) > 0.7

def process_images():
    for filename in os.listdir(train_path):
        if (filename.startswith("2")): #and (not filename.startswith('6175')):
            print(filename)
            image_path = os.path.join(train_path, filename)
            imageId, _ = os.path.splitext(filename)
            original_image = Image.open(image_path)
            fullres_dimensions = original_image.size
            
            thumb_path = os.path.join(thumbnail_path, f"{imageId}_thumbnail.png")
            tsa = False
            if not os.path.exists(thumb_path):
                tsa = True

            sample_directory = os.path.join(output_path, f"sample_{imageId}")
            os.makedirs(sample_directory, exist_ok=True)

            all_entries = os.listdir(sample_directory)
            files = [entry for entry in all_entries if os.path.isfile(os.path.join(sample_directory, entry))]
            valid_patches = len(files)
            coords = []
            if tsa == False and valid_patches==0:
                coords = get_tiles_coordinates(thumb_path, fullres_dimensions)
                coords = filter_close_coordinates(coords, 250)
                print(len(coords))
                for x, y in coords:
                    if valid_patches >= 100:
                        break
                    cropped_img = original_image.crop((x, y, x + 250, y + 250))
                    if is_valid_patch(cropped_img):
                        patch_filename = f"{imageId}_{valid_patches}.png"
                        cropped_img.save(os.path.join(sample_directory, patch_filename))
                        valid_patches += 1
            print("a", valid_patches)
            indicesx = list(range(fullres_dimensions[0] - 250))
            indicesy = list(range(fullres_dimensions[1] - 250))
            sampled_indicesx = random.sample(indicesx, 2500)
            sampled_indicesy = random.sample(indicesy, 2500)

            for random_x, random_y in list(zip(sampled_indicesx, sampled_indicesy)):
                if valid_patches >= 100:
                        break
                random_coord = (random_x, random_y)
                if random_coord not in coords:  # Ensure no duplicates
                    oriLen = len(coords)
                    coords.append(random_coord)
                    if len(filter_close_coordinates(coords, 250)) > oriLen:
                        cropped_img = original_image.crop((random_x, random_y, random_x + 250, random_y + 250))
                        if is_valid_patch(cropped_img):
                            patch_filename = f"{imageId}_{valid_patches}.png"
                            cropped_img.save(os.path.join(sample_directory, patch_filename))
                            valid_patches += 1
                            print(valid_patches)
                    else:
                        coords.pop()

# Run the process
process_images()
