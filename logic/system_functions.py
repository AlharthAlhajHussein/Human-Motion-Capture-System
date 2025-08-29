import cv2
import json
import os
from PIL import Image, ExifTags
import numpy as np

# ===== Mapping of required landmarks to their indices in MediaPipe =====
landmark_mapping = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_up_leg': 23,
    'right_up_leg': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

# ===== Define connections between keypoints for visualization =====
connections = [
    ('head', 'neck'),
    ('neck', 'spine'),
    ('spine', 'hip'),
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('hip', 'left_up_leg'),
    ('hip', 'right_up_leg'),
    ('left_up_leg', 'left_knee'),
    ('right_up_leg', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
]

# ===== Extract 2D landmarks from MediaPipe results for case 2D keypoints =====
def extract_2D_landmarks(results):
    """Extract 2D landmarks from MediaPipe results.
    
    Args:
        results (MediaPipe results): the results from the MediaPipe model
    
    Returns:
        landmark_2D_list (list): the list of dictionaries of 2D landmarks
    """

    landmark_2D_list = []
    
    for idx in range(33):
        landmark = results.pose_landmarks.landmark[idx]
        if landmark:
            landmark_2D_list.append({
                'x': landmark.x,
                'y': landmark.y,
            })
        else:
            print(f"Landmark {idx} is not found")

    return landmark_2D_list

# ===== Extract 3D landmarks from MediaPipe results for case 3D (fixed) keypoints =====
def extract_3D_landmarks_fixed(results):
    """Extract 3D landmarks from MediaPipe results.

    Args:
        results (MediaPipe results): the results from the MediaPipe model

    Returns:
        landmark_3D_list (list): the list of dictionaries of 3D landmarks
    """
    
    landmark_3D_list = []
    for idx in range(33):
        landmark = results.pose_world_landmarks.landmark[idx]
        if landmark:
            landmark_3D_list.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
            })
        else:
            print(f"Landmark {idx} is not found")
    
    return landmark_3D_list

# ===== Extract x, y 2D and z 3D landmarks from MediaPipe results for case 3D (movable) keypoints =====
def extract_3D_landmarks_movable(results):
    """Extract x, y 2D and z 3D landmarks from MediaPipe results for case 3D (movable) keypoints.

    Args:
        results (MediaPipe results): the results from the MediaPipe model

    Returns:
        landmark_x_y_2D_z_3D_list (list): the list of dictionaries of x, y 2D and z 3D landmarks
    """
    
    landmark_x_y_2D_z_3D_list = []
    
    for idx in range(33):
        landmark = results.pose_landmarks.landmark[idx]
        if landmark:
            landmark_x_y_2D_z_3D_list.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
            })
        else:
            print(f"Landmark {idx} is not found")
    
    return landmark_x_y_2D_z_3D_list
                       
# ===== Calculate extra landmarks =====
def calculate_extra_landmarks(landmark_list):
    """Calculate extra landmarks (Head, Neck, Spine, Hip) for all cases.
    
    Args:
        landmark_list (list): the list of dictionaries of landmarks
    
    Returns:
        extra_landmarks (list): the list of dictionaries of extra landmarks
    """
    
    # Extract specific landmarks to calculate the extra points
    left_shoulder = landmark_list[11] 
    right_shoulder = landmark_list[12]
    left_hip = landmark_list[23]     # or left Up leg
    right_hip = landmark_list[24]    # or right Up leg
    left_mouth = landmark_list[9]
    right_mouth = landmark_list[10]
    nose = landmark_list[0]
    left_ear = landmark_list[7]
    right_ear = landmark_list[8]
    
    
    # Calculate Spine coordinates
    spine = {
        'x': (left_hip['x'] + right_hip['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
        'y': (left_hip['y'] + right_hip['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
        'name': 'spine'
    }
    
    # Calculate Hip coordinates
    hip = {
        'x': (left_hip['x'] + right_hip['x'] + spine['x']) / 3.0,
        'y': (left_hip['y'] + right_hip['y'] + spine['y']) / 3.0,
        'name': 'hip'
    }

    # Calculate Neck coordinates
    neck = {
        'x': (left_mouth['x'] + right_mouth['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
        'y': (left_mouth['y'] + right_mouth['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
        'name': 'neck'
    }

    # Calculate Head coordinates
    head = {
        'x': (nose['x'] + left_ear['x'] + right_ear['x']) / 3.0,
        'y': (nose['y'] + left_ear['y'] + right_ear['y']) / 3.0,
        'name': 'head'
    }
    
    # Check if the z-coordinates are available in the landmark list
    if all('z' in joint for joint in [left_hip, right_hip, left_shoulder, right_shoulder, nose, left_mouth, right_mouth]):
        # Calculate the z-coordinates of the extra points
        spine['z'] = (left_hip['z'] + right_hip['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0
        hip['z'] = (left_hip['z'] + right_hip['z'] + spine['z']) / 3.0
        neck['z'] = (left_mouth['z'] + right_mouth['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0
        head['z'] = (nose['z'] + left_ear['z'] + right_ear['z']) / 3.0
        
    return [hip, spine, neck, head]
     
# ===== Get required landmarks =====
def get_required_landmark(all_landmarks, extra_landmarks):
    """ Get required landmarks form all landmarks (33) and extra landmarks (4) to obtain the only 16 required landmarks.
    
    Args:
        all_landmarks (list): the list of dictionaries of all landmarks (33)
        extra_landmarks (list): the list of dictionaries of extra landmarks (4)
    
    Returns:
        required_landmarks (dict): the dictionary of 16 required landmarks
    """

    required_landmarks = {}
    
    for name, idx in landmark_mapping.items():
        required_landmarks[name] = all_landmarks[idx]
        required_landmarks[name]['name'] = name
    
    for landmark in extra_landmarks:
        required_landmarks[landmark['name']] = landmark
    
    return required_landmarks

# ===== Denormalize landmarks for case 2D and 3D (movable) keypoints =====
def denormalize_landmarks(image, landmark_dicts):
    """Denormalize landmarks to the original image size for the case of 2D and 3D (movable) keypoints.
    
    Args:
        image (cv2.Mat): the original image
        landmark_dicts (dict): the dictionary of landmarks
    """
    height, width = image.shape[:2]
    
    for name, landmark in landmark_dicts.items():
        landmark['x'] = round(landmark['x'] * width, 3)
        landmark['y'] = round(landmark['y'] * height, 3)

# ===== Project landmarks for all cases =====
def project_landmarks(image, landmark_dicts):
    """Project landmarks on the image.
    
    Args:
        image (cv2.Mat): the original image
        landmark_dicts (dict): the dictionary of landmarks
    """
    
    # Draw landmarks
    for name, landmark in landmark_dicts.items():
        x = int(landmark['x'])
        y = int(landmark['y'])
        cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
        # cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
# ===== Project skeleton =====
def project_skeleton(image, landmark_dicts):
    """Project skeleton on the image.
    
    Args:
        image (cv2.Mat): the original image
        landmark_dicts (dict): the dictionary of landmarks
    """
    
    # Draw skeleton
    for start_point, end_point in connections:
        
        # Check if the start and end points are in the landmark dictionary
        if start_point in landmark_dicts and end_point in landmark_dicts:
            
            # Get the start and end points from the landmark dictionary
            start = landmark_dicts[start_point]
            end = landmark_dicts[end_point]
            
            # Convert the normalized coordinates to pixel coordinates
            start_x = int(start['x'])
            start_y = int(start['y'])
            end_x = int(end['x'])
            end_y = int(end['y'])
            
            # Draw the connection between the start and end points
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
# ===== Project special values =====
def project_special_values(image, landmark_dicts, scaled_landmarks_dicts=None):
    """Project 6 special values the x for left and right wrists also the y for left and right ankles, head also the x&y for hip.
    
    Args:
        image (cv2.Mat): the original image
        landmark_dicts (dict): the dictionary of landmarks for the case of 2D or 3D (movable) keypoints
        scaled_landmarks_dicts (dict): the dictionary of scaled landmarks for the case of 3D (fixed) keypoints
    """
    
    # Define the names of the keypoints to draw the values of
    names_Y = ['head', 'left_ankle', 'right_ankle', 'hip']  # draw the Y values of these keypoints
    names_X = ['left_wrist', 'right_wrist', 'hip']          # draw the X values of these keypoints
    
    # For the case of 2D and 3D (movable) keypoints
    if scaled_landmarks_dicts is None:
        for name, landmark in landmark_dicts.items():
            if name in names_Y:
                y = int(landmark['y'])
                x = int(landmark['x'])
                cv2.putText(image, f"y={y}", (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if name in names_X:
                x = int(landmark['x'])
                y = int(landmark['y'])
                if name == 'hip':
                    cv2.putText(image, f"x={x}", (x, y-16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(image, f"x={x}", (x, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        # For the case of 3D (fixed) keypoints, we need to store the 2D positions of the keypoints
        store_2d_positions = {}
        
        for name, landmark in landmark_dicts.items():
            if name in names_Y or name in names_X:
                y = int(landmark['y'])
                x = int(landmark['x'])
                store_2d_positions[name] = (x, y)
                
        for name, landmark in scaled_landmarks_dicts.items():
            if name in names_Y:
                y_value = round(landmark['y'], 2)
                x, y = store_2d_positions[name]
                cv2.putText(image, f"y={y_value}", (x+6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if name in names_X:
                x_value = round(landmark['x'], 2)
                x, y = store_2d_positions[name]
                if name == 'hip':
                    cv2.putText(image, f"x={x_value}", (x+6, y-16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(image, f"x={x_value}", (x, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# ===== Calculate scaling parameters for case 3D (fixed) keypoints =====
def calculate_scaling_params(landmark_dicts):
    """Calculate the scaling parameters for the landmarks for the case of 3D (fixed) keypoints.

    Args:
        landmark_dicts (dict): the dictionary of landmarks for the case of 3D (fixed) keypoints

    Returns:
        scaling_params (tuple): scaling parameters (w1, w2, lowest_point)
    """
    
    head_dic = landmark_dicts['head']
    LW_dic = landmark_dicts['left_wrist']
    RW_dic = landmark_dicts['right_wrist']
    LA_dic = landmark_dicts['left_ankle']
    RA_dic = landmark_dicts['right_ankle']
    
    # Get the lowest point (feet)
    lowest_point = max(LA_dic['y'], RA_dic['y'])
    
    # Calculate w1 (width scale).
    w1 = 2.9 / (abs(LW_dic['x']) + abs(RW_dic['x']))
    
    # Calculate w2 (height scale).
    w2 = 3.1 / (abs(head_dic['y']) + abs(lowest_point))

    return (w1, w2, lowest_point)

# ===== Scale landmarks for case 3D (fixed) keypoints =====
def scale_landmarks(landmark_dicts, w1, w2, lowest_point):
    """Scale the landmarks to the original character size (2.9 units width and 3.1 units height) for the case of 3D (fixed) keypoints.

    Args:
        landmark_dicts (dict): the dictionary of landmarks for the case of 3D (fixed) keypoints
        w1 (float): the scaling parameter for the width
        w2 (float): the scaling parameter for the height
        lowest_point (float): the lowest point of the landmarks
    """ 
    
    for name, landmark in landmark_dicts.items():
        landmark['x'] = round(landmark['x'] * w1, 3)
        landmark['y'] = round((lowest_point - landmark['y']) * w2, 3)
        landmark['z'] = round(landmark['z'] * 2.5, 3)
    
# ===== Adjust landmarks for case 3D (movable) keypoints =====
def adjust_landmarks(image,landmark_dicts):
    """Adjust the landmarks to fit with the original character size for the case of 3D (movable) keypoints.
    
    Args:
        image (cv2.Mat): the original image
        landmark_dicts (dict): the dictionary of landmarks for the case of 3D (movable) keypoints
    """
    
    height = image.shape[0]
    
    LA_y = landmark_dicts['left_ankle']['y']
    RA_y = landmark_dicts['right_ankle']['y']
    lowest_point = (height - max(LA_y, RA_y)) / 111
    
    
    for name, landmark in landmark_dicts.items():
        landmark['x'] = round(landmark['x'] / 111, 3)
        landmark['y'] = round(((height - landmark['y']) / 111) - lowest_point, 3)
        landmark['z'] = round(landmark['z'] * 2.5, 3)    

# ===== Save landmarks =====
def save_keypoints(landmark, file_name):
    """Save the landmarks to a file.
    
    Args:
        landmark (list): the list of dictionaries for each frame containing the landmarks
        file_name (str): the name of the file
    """
    
    if file_name.endswith('.json'):
        file_path = os.path.join('outputs', 'keypoints', file_name)
    else:
        file_path = os.path.join('outputs', 'keypoints', file_name + '.json')

    # Save the landmarks to a json file
    with open(file_path, 'w') as f:
        json.dump(landmark, f, indent=4)

    print(f"Video landmarks saved to {file_path}")
    
# ===== Save processed image =====
def save_processed_image(image, file_name, size_str):
    """
    Saves the processed image to a file, resizing it if necessary.

    Args:
        image (cv2.Mat): The processed image data.
        file_name (str): the name of the file
        size_str (str): A string representing the desired output size (e.g., "1920x1080").
    """
    
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.bmp'):
        output_path = os.path.join('outputs', 'images', file_name)
    else:
        output_path = os.path.join('outputs', 'images', file_name + '.png')

    output_image = image.copy()

    # Resize the image if a specific size is requested
    if size_str != "Original":
        try:
            width, height = map(int, size_str.split('x'))
            output_image = cv2.resize(output_image, (width, height))
        except ValueError:
            print(f"Warning: Invalid size format '{size_str}'. Saving with original dimensions.")

    # Save the final image to disk
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to: {output_path}")
    
# ===== Load image with orientation =====
def load_image_with_orientation(image_path):
    """
    Loads an image, corrects its orientation based on EXIF data, and returns it
    as an OpenCV-compatible BGR NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The correctly oriented image in BGR format, or None if loading fails.
    """
    try:
        img = Image.open(image_path)

        # A more robust way to get EXIF data
        try:
            exif_data = img._getexif()
            if exif_data is None:
                exif_data = {}
        except Exception:
            exif_data = {} # If _getexif fails for any reason

        exif = {
            ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in ExifTags.TAGS
        }
        orientation = exif.get('Orientation', 1) # Default to 1 (normal)

        # Apply rotation/transposition based on orientation tag
        if orientation == 2:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 4:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.rotate(270, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

        # Convert from PIL's format to OpenCV's BGR format
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        image_np_rgb = np.array(img)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        
        return image_np_bgr

    except Exception as e:
        print(f"Error during orientation correction: {e}. Falling back to standard image loading.")
        return cv2.imread(image_path)
    
    
    


