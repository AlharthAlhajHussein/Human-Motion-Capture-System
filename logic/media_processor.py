import cv2
import numpy as np
import mediapipe as mp
from logic.system_functions import (

    extract_2D_landmarks,
    calculate_extra_landmarks,
    get_required_landmark,
    denormalize_landmarks,
    project_landmarks,
    project_skeleton,
    save_keypoints,
    save_processed_image,
    project_special_values
)
from logic.system_functions import load_image_with_orientation
class MediaProcessor:
    """
    Handles the entire media processing pipeline for images and videos.
    """ 
    def __init__(self, model_complexity=1):
        """
        Initializes the MediaProcessor and both MediaPipe Pose models.
        """
        # Create a Pose instance for static images
        self.image_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=0.5
        )
        # Create a separate Pose instance for video streams
        self.video_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5     
        )

    # ===== Process image =====
    def process_image(self, image_path, plot_landmarks, plot_skeleton, save_landmarks, landmarks_filename, save_image, output_size_str, image_filename, save_image_black_background, image_black_background_filename):
        """
        Runs the full pipeline on a single image.

        Args: 
            image_path (str): The full path to the image file.
            plot_landmarks (bool): Whether to draw landmarks on the image.
            plot_skeleton (bool): Whether to draw the skeleton on the image.
            save_landmarks (bool): Whether to save the landmark data to a file.
            landmarks_filename (str): The name of the file to save to. Can be None.
            save_image (bool): Whether to save the processed image.
            output_size_str (str): The desired output size for the image.
            image_filename (str): The name of the file to save to. Can be None.
            save_image_black_background (bool): Whether to save the image with black background.
            image_black_background_filename (str): The name of the file to save to. Can be None.
        Returns:
            processed_image: The image with landmarks and skeleton drawn on it (as a NumPy array).
            Returns None if processing fails.
        """
        try:
            
            image = load_image_with_orientation(image_path)
            
            if image is None:
                # The error message is already printed by the loading function
                return None
                
            # --- MediaPipe Processing ---
            # Process the image and find pose landmarks using the image model
            results = self.image_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                print("Warning: No pose landmarks detected in the image.")
                return image # Return the original image if no landmarks are found

            # --- Function Call Pipeline ---
            # 1. Extract 2D landmarks
            landmarks_2d = extract_2D_landmarks(results)

            # 2. Calculate extra landmarks (head, neck, etc.)
            extra_landmarks_list = calculate_extra_landmarks(landmarks_2d)

            # 3. Get the final 16 required landmarks
            required_landmarks_dict = get_required_landmark(landmarks_2d, extra_landmarks_list)

            # 4. De-normalize landmark coordinates
            denormalize_landmarks(image, required_landmarks_dict)

            # 5. Project landmarks onto the image (conditionally)
            if plot_landmarks:
                project_landmarks(image, required_landmarks_dict)

            # 6. Project the skeleton connections (conditionally)
            if plot_skeleton:
                project_skeleton(image, required_landmarks_dict)

            # 7. Save the landmark data to a file (conditionally)
            if save_landmarks and landmarks_filename:
                save_keypoints(required_landmarks_dict, landmarks_filename)

            # 8. Save the processed image (conditionally)
            if save_image and output_size_str:
                save_processed_image(image, image_filename, output_size_str)
            
            # 9. Save the processed image with black background (conditionally)
            if save_image_black_background and image_black_background_filename:
                black_background_image = np.zeros_like(image)
                if plot_landmarks:
                    project_landmarks(black_background_image, required_landmarks_dict)
                if plot_skeleton:
                    project_skeleton(black_background_image, required_landmarks_dict)
                # 10. Save the processed image with black background
                save_processed_image(black_background_image, image_black_background_filename, output_size_str)

            # return image by default 
            if save_image and image_filename:
                return image
            elif save_image_black_background and image_black_background_filename:
                return black_background_image
            else:
                return image

        except Exception as e:
            print(f"An error occurred during image processing: {e}")
            return None

    # ===== Process video frame by frame =====
    def process_video_frame(self, frame, plot_landmarks, plot_skeleton, plot_values, save_video_black_background):
        """
        Processes a single video frame.

        Args:
            frame: The video frame (NumPy array).
            plot_landmarks (bool): Whether to draw landmarks.
            plot_skeleton (bool): Whether to draw the skeleton.
            plot_values (bool): Whether to draw the values for (wrists, head, ankles).
            save_video_black_background (bool): Whether to save the video with black background.
        Returns:
            A tuple containing:
            - The processed frame (NumPy array).
            - A dictionary of the final 16 required landmarks, or None.
        """
        # Process the frame and find pose landmarks using the video model
        results = self.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        required_landmarks_dict = None
        if results.pose_landmarks:
            landmarks_2d_list = extract_2D_landmarks(results)
            extra_landmarks_list = calculate_extra_landmarks(landmarks_2d_list)
            required_landmarks_dict = get_required_landmark(landmarks_2d_list, extra_landmarks_list)
            denormalize_landmarks(frame, required_landmarks_dict)
            
            if plot_landmarks:
                project_landmarks(frame, required_landmarks_dict)
            if plot_skeleton:
                project_skeleton(frame, required_landmarks_dict)
            if plot_values:
                project_special_values(frame, required_landmarks_dict)
            if save_video_black_background:
                black_background_frame = np.zeros_like(frame)
                if plot_landmarks:
                    project_landmarks(black_background_frame, required_landmarks_dict)
                if plot_skeleton:
                    project_skeleton(black_background_frame, required_landmarks_dict)
                if plot_values:
                    project_special_values(black_background_frame, required_landmarks_dict)
                return frame, required_landmarks_dict, black_background_frame
            
        return frame, required_landmarks_dict, None

    # ===== Save video landmarks =====
    def save_video_landmarks(self, all_frame_landmarks, landmarks_filename):
        """Saves all collected landmarks from a video to a file."""
        if all_frame_landmarks and landmarks_filename:
            save_keypoints(all_frame_landmarks, landmarks_filename)

    # ===== Get video rotation =====
    def get_video_rotation(self, cap):
        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))        
        if orientation == 90:
            rotation_code= cv2.ROTATE_90_CLOCKWISE
        elif orientation == -90:
            rotation_code= cv2.ROTATE_90_COUNTERCLOCKWISE
        elif orientation == 180:
            rotation_code= cv2.ROTATE_180
        else:
            rotation_code= None
        return rotation_code
    
    # ===== Close =====
    def close(self):
        """Clean up the MediaPipe Pose object."""
        self.image_pose.close()
        self.video_pose.close() 