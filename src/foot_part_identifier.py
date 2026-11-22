import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Fixed Variables for Testing
# ===========================
day_index = 1
pnt = "Patient 1"
output_dir = "foot_images_folder"

# Directories #
## MAT-Files ##
mat_folder_path ="./Data/Temp Data"

# ===========================
# Helper Functions
# ===========================

def trim_empty_columns(img):
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

def find_width(two_dim_array):
    if not two_dim_array.any():
        return [], -1, -1

    max_count = 0
    widest_array = []
    index_of_widest_array = -1
    midpoint_index_of_widest_array = -1

    for index, arr in enumerate(two_dim_array):
        # Get the indices of all non-NaN values
        non_nan_indices = np.where(~np.isnan(arr))[0]
        current_count = len(non_nan_indices)
        
        # Check if this count is greater than the current maximum
        if current_count > max_count:
            max_count = current_count
            widest_array = arr
            index_of_widest_array = index
            
            # Calculate the midpoint index of the non-NaN values
            if current_count > 0:
                midpoint_index_of_widest_array = non_nan_indices[current_count // 2]
            else:
                midpoint_index_of_widest_array = -1
            
    return widest_array, index_of_widest_array, midpoint_index_of_widest_array

def horizontal_split_by_percentage(two_dim_array, percentage):
    
    if isinstance(two_dim_array, list):
        two_dim_array = np.array(two_dim_array)
    
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")
    
    last_index = len(two_dim_array) - 1
    first_index = 0
    
    while np.all(np.isnan(two_dim_array[first_index])) and (first_index != last_index): 
        first_index+=1
    while np.all(np.isnan(two_dim_array[last_index])) and not(last_index < 0): 
        last_index-=1    
    
    # Calculate the total height of the foot
    foot_height = last_index - first_index + 1
    
    # Calculate the split row index
    split_row_index = first_index + int(foot_height * percentage)
    
    # Use NumPy's slicing to create the top and bottom portions
    top_portion = two_dim_array[:split_row_index, :]
    bottom_portion = two_dim_array[split_row_index:, :]

    return top_portion, bottom_portion

def vertical_split_by_percentage(two_dim_array, percentage):
    if isinstance(two_dim_array, list):
        two_dim_array = np.array(two_dim_array)
    
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")
    
    widest_array, index_of_widest_array, midpoint_index_of_widest_array = find_width(two_dim_array)
    
    print(widest_array)
    
    # Create a boolean mask of where values are NOT NaN.
    # '~' is the logical NOT operator.
    non_nan_mask = ~np.isnan(widest_array)
    
    non_nan_indices = np.where(non_nan_mask)[0]

    # Check if any non-NaN values were found and get the first index
    if non_nan_indices.size > 0:
        first_non_nan_index = non_nan_indices[0]
        
        # Sum the boolean mask. In NumPy, True is treated as 1 and False as 0.
        # The sum of the mask gives the total count of non-NaN values.
        length_of_non_nan_values = np.sum(non_nan_mask)
        
        split_index_by_percentage = first_non_nan_index + int(length_of_non_nan_values * percentage)
        
        left_half = two_dim_array[:, :split_index_by_percentage]
        
        # The right side is from the foot split index to the end of the original image
        right_half = two_dim_array[:, split_index_by_percentage:]
        
        return left_half, right_half
    else:
        print("No non-NaN values found in the array.")
        empty_array = np.array([], dtype=np.int64)
        return empty_array, empty_array
    
def segment_foot(foot_arr, foot_target):
    
    if foot_target == "Indirect_plantar":
        top, bottom = horizontal_split_by_percentage(foot_arr, 0.65)
        heel, mid_foot = horizontal_split_by_percentage(top, 0.4)
    else:
        top, bottom = horizontal_split_by_percentage(foot_arr, 0.65)
        heel, mid_foot = horizontal_split_by_percentage(top, 0.4)

    return heel, mid_foot, bottom

# Out-Dated Functions

def initial_vertical_split(two_dim_array, mid_point_index):
    # Check if the input is a list and convert it to a NumPy array
    if isinstance(two_dim_array, list):
        two_dim_array = np.array(two_dim_array)
        
    # Ensure the input is a 2D NumPy array after the conversion attempt
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")
    
    # Use NumPy's slicing to create the two halves
    # Note: We still slice from the original image to maintain the
    # original dimensions for the background.
    # The left side is from the original start to the foot midpoint
    left_half = two_dim_array[:, :mid_point_index]
    
    # The right side is from the foot midpoint to the end of the original image
    right_half = two_dim_array[:, mid_point_index:]
    
    return left_half, right_half

def initial_horizontal_split(two_dim_array, width_array_index):
    # Check if the input is a list and convert it to a NumPy array.
    if isinstance(two_dim_array, list):
        image_data = np.array(two_dim_array)

    # Ensure the input is a 2D NumPy array after the conversion attempt
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")

    # Use NumPy's slicing to create the two halves
    top_half = two_dim_array[:width_array_index, :]
    bottom_half = two_dim_array[width_array_index:, :]

    return top_half, bottom_half

def segment_foot_old(foot_arr):
    _, index_of_widest_array, midpoint_index_of_widest_array = find_width(foot_arr)
    top_right, bottom_right = initial_horizontal_split(foot_arr, index_of_widest_array)
    
    right_top_left, right_top_right = initial_vertical_split(top_right, midpoint_index_of_widest_array)
    right_bottom_left, right_bottom_right = initial_vertical_split(bottom_right, midpoint_index_of_widest_array)
    
    return right_top_left, right_top_right, right_bottom_left, right_bottom_right
    
# ---------------------------

if __name__ == "__main__":

    ## Testing With Healthy 1 ##
    mat_pnt_one = scipy.io.loadmat(mat_folder_path + "/gz1.mat")

    # Working on right foot for now #
    right_crop = mat_pnt_one["Dorsal_Right_crop"]
    scan_right = right_crop[day_index, 0]
    img_right = trim_empty_columns(np.where(scan_right == 0, np.nan, scan_right))
    
    left_crop = mat_pnt_one["Indirect_plantar_Left_crop"]
    scan_left = left_crop[day_index, 0]
    img_left = trim_empty_columns(np.where(scan_left == 0, np.nan, scan_left))
    
    # TESTING WITH THE RIGHT FOOT RIGHT NOW NEW #
    heel_split, mid_foot, upper_foot = segment_foot(img_right, "Dorsal")
    gap_horizontal = np.full((3, heel_split.shape[1]), np.nan)
    
    combined = np.vstack((heel_split, gap_horizontal, mid_foot, gap_horizontal, upper_foot))
    
    # OLD CODE TESTING #
    # right_top_left, right_top_right, right_bottom_left, right_bottom_right = segment_foot_old(img_right)
    
    # gap_vertical_top = np.full((right_top_right.shape[0], 2), np.nan)
    # combined_top = np.hstack((right_top_left, gap_vertical_top, right_top_right))
    
    # gap_vertical_bottom = np.full((right_bottom_left.shape[0], 2), np.nan)
    # combined_bottom = np.hstack((right_bottom_left, gap_vertical_bottom, right_bottom_right))
    
    # gap_horizontal = np.full((2, combined_top.shape[1]), np.nan)
    
    # combined_top_bottom = np.vstack((combined_top, gap_horizontal, combined_bottom))
    
    
    
    # combined_left_right = np.hstack((left_side, gap_vertical, right_side))
    # combined_top_bottom = np.vstack((heel_split, gap_horizontal, bottom))
    # left_foot_padding = np.full((4, img_left.shape[1]), np.nan)
    # padded_left_foot = np.vstack((left_foot_padding, img_left))
    
    # left_right_gap = np.full((combined_top_bottom.shape[0], 5), np.nan)
    # combined_left_right_feet = np.hstack((combined_top_bottom, left_right_gap, padded_left_foot))

    # ==========================
    # Plot and save
    # ==========================

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(combined, cmap="hot", interpolation='nearest')
    ax.axis("off")
    ax.set_title(f"Day {day_index+1}: Foot for {pnt} Left Side")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.show()

    # plt.savefig(os.path.join(output_dir, f"foot_day{day_index+1}_left_side"),
    #             dpi=300, bbox_inches="tight")
    # plt.close()

    # os.makedirs(output_dir, exist_ok=True)
    # scipy.io.savemat(os.path.join(output_dir, f"{pnt}{day_index+1}_left_side.mat"),
    #                 {"foot_image": left_side})

    # print(f"Saved combined PNG in '{output_dir}' ({pnt})_left_side")