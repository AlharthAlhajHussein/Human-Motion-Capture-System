from PyQt6.QtCore import QObject, pyqtSignal, QCoreApplication
from logic.media_processor import MediaProcessor
import cv2
import os
import requests
import numpy as np
import matplotlib
from logic.system_functions import (
    extract_2D_landmarks,
    extract_3D_landmarks,
    calculate_extra_landmarks,
    get_required_landmark,
    denormalize_landmarks,
    project_landmarks,
    project_skeleton,
    project_special_values,
    save_keypoints,
    get_depth_for_hip_keypoint,
    shifting_keypoints_with_z_value,
    get_norm_x_for_hip,
    shifting_keypoints_with_x_value,
)
from logic.websocket_server import KeypointServer
import torch
from logic.depth_anything_v2.dpt import DepthAnythingV2

class Worker(QObject):
    """
    The worker class that handles all long-running processing tasks.
    It lives on a separate thread.
    """
    # Signals that the worker can emit
    image_finished = pyqtSignal(object) # The 'object' will be the processed NumPy image array
    video_finished = pyqtSignal(str)    # The 'str' will be a completion message
    new_frame_ready = pyqtSignal(object) # Signal to send a new processed frame
    error = pyqtSignal(str)             # The 'str' will be an error message

    def __init__(self, encoder='vits', checkpoint_path=None):
        super().__init__()
        # The MediaProcessor is now owned by the worker
        self.media_processor = MediaProcessor()
        self.is_running = False # Flag to control the processing loop
        
        # Device selection: CUDA > MPS > CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Model configurations for different encoders
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.encoder = encoder
        
        # Initialize the model
        self.model = DepthAnythingV2(**self.model_configs[encoder])
        
        # Load model weights
        if checkpoint_path is None:
            checkpoint_path = f'logic/checkpoints/depth_anything_v2_{encoder}.pth'
        
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from: {checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            print("Please make sure you have downloaded the model weights.")
            exit(1)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device).eval()
        
        # Initialize colormap for depth visualization (Spectral_r gives nice colored depth maps)
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
    def switch_mediapipe_model(self, model_comp):
        """Switches the model complexity
        Args:
            model_complixity (int): The complexity of the model 1 light or 2 heavy
        """
        self.media_processor = MediaProcessor(model_complexity=model_comp)
    
    def switch_depth_anything_model(self, model_size):
        """
        Switches the depth model size and reinitializes the model on GPU
        
        Args:
            model_size (str): The size of the depth model ('vits', 'vitb', 'vitl', 'vitg')
        """
        
        self.stop()
        
        try:
            print(f"Switching Depth Anything model to: {model_size}")
            
            # Validate model size
            if model_size not in self.model_configs:
                raise ValueError(f"Invalid model size '{model_size}'. Valid options: {list(self.model_configs.keys())}")
            
            # Clear current model from GPU memory to prevent memory issues
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("Cleared GPU cache")
            
            # Update encoder
            self.encoder = model_size
            
            # Initialize new model with the selected configuration
            self.model = DepthAnythingV2(**self.model_configs[model_size])
            
            # Load model weights for the new size
            checkpoint_path = f'logic/checkpoints/depth_anything_v2_{model_size}.pth'
            
            try:
                print(f"Loading checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                print(f"Model weights loaded successfully from: {checkpoint_path}")
            except FileNotFoundError:
                error_msg = f"Error: Checkpoint file not found at {checkpoint_path}"
                print(error_msg)
                print(f"Please make sure you have downloaded the {model_size} model weights.")
                # Emit error signal if available
                if hasattr(self, 'error'):
                    self.error.emit(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            # Move model to device (GPU/CPU) and set to evaluation mode
            self.model = self.model.to(self.device).eval()
            
            print(f"Successfully switched to {model_size} model on device: {self.device}")
            
            # Print model info
            model_info = {
                'vits': 'Small Model (Fastest, Lower Accuracy)',
                'vitb': 'Base Model (Balanced Speed/Accuracy)', 
                'vitl': 'Large Model (Slower, Higher Accuracy)',
                'vitg': 'Giant Model (Slowest, Highest Accuracy)'
            }
            
            print(f"Active model: {model_info.get(model_size, model_size)}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to switch depth model: {str(e)}"
            print(error_msg)
            if hasattr(self, 'error'):
                self.error.emit(error_msg)
            return False
        
    def process_image(self, image_path, plot_landmarks, plot_skeleton, save_landmarks, landmark_filename, save_image, output_size_str, processed_image_filename, save_image_black, processed_image_black_background_filename):
        """A slot that processes the image and emits a signal when done."""
        try:
            result = self.media_processor.process_image(
                image_path=image_path,
                plot_landmarks=plot_landmarks,
                plot_skeleton=plot_skeleton,
                save_landmarks=save_landmarks,
                landmarks_filename=landmark_filename,
                save_image=save_image,
                output_size_str=output_size_str,
                image_filename=processed_image_filename,
                save_image_black_background=save_image_black,
                image_black_background_filename=processed_image_black_background_filename
            )
            self.image_finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def process_video(self, video_path, plot_landmarks, plot_skeleton, plot_values, save_landmarks, save_video, landmark_filename, video_filename, save_video_black_background, video_black_background_filename):
        """A slot that processes the video and emits a signal when done."""
        self.is_running = True # Set the running flag to True
        try:
            
            # integrate the is_running flag and the new_frame_ready signal.
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")

            # Get rotation and corrected dimensions
            rotation_code = self.media_processor.get_video_rotation(cap)

            ret, frame = cap.read()
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
            

            writer = None
            writer_black_background = None
            
            # Setup video writer if saving
            if save_video and video_filename:
                writer = self.init_writer(cap, video_filename, frame)
            
            if save_video_black_background and video_black_background_filename:
                frame_black_background = np.zeros_like(frame)
                writer_black_background = self.init_writer(cap, video_black_background_filename, frame_black_background)
            
            all_frame_landmarks = []

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                
                # Apply rotation if the video file has orientation metadata
                if rotation_code is not None:
                    frame = cv2.rotate(frame, rotation_code)

                # The actual processing logic is in MediaProcessor, but we call it from here
                processed_frame, landmarks_dict, black_background_frame = self.media_processor.process_video_frame(frame, plot_landmarks, plot_skeleton, plot_values, save_video_black_background)

                if landmarks_dict and save_landmarks:
                    all_frame_landmarks.append(landmarks_dict)

                if save_video and video_filename:
                    self.new_frame_ready.emit(processed_frame)
                elif save_video_black_background and video_black_background_filename and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(processed_frame)

                if writer: 
                    writer.write(processed_frame)
                if writer_black_background:
                    writer_black_background.write(black_background_frame)

                QCoreApplication.processEvents() # Process events to remain responsive to stop signals

            # Post-loop saving and cleanup
            if save_landmarks and landmark_filename:
                self.media_processor.save_video_landmarks(all_frame_landmarks, landmark_filename)
            
            cap.release()
            if writer:
                writer.release()
                print(f"Video saved to {video_filename}")
            if writer_black_background:
                writer_black_background.release()
                print(f"Video with black background saved to {video_black_background_filename}")
            
            
            completion_message = "Video processing stopped by user." if not self.is_running else "Video processing complete."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))

    def process_webcam(self, plot_landmarks, plot_skeleton, plot_values, save_landmarks, save_video, landmark_filename, video_filename, save_video_black_background, video_black_background_filename):
        """A slot that processes the webcam feed."""
        self.is_running = True
        try:
            cap = cv2.VideoCapture(0) # 0 is the default camera
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam.")

            writer = None
            writer_black_background = None
            
            if save_video and video_filename:
                writer = self.init_writer_webcam(cap, video_filename, 15)
            
            if save_video_black_background and video_black_background_filename:
                writer_black_background = self.init_writer_webcam(cap, video_black_background_filename, 15)
            
            all_frame_landmarks = []

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("Failed to capture frame from webcam.")
                    break

                processed_frame, landmarks_dict, black_background_frame = self.media_processor.process_video_frame(frame, plot_landmarks, plot_skeleton, plot_values, save_video_black_background)

                if landmarks_dict and save_landmarks:
                    all_frame_landmarks.append(landmarks_dict)

                if save_video and video_filename:
                    self.new_frame_ready.emit(processed_frame)
                elif save_video_black_background and video_black_background_filename and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(processed_frame)

                if writer:
                    writer.write(processed_frame)
                if writer_black_background:
                    writer_black_background.write(black_background_frame)
                QCoreApplication.processEvents()

            if save_landmarks and landmark_filename:
                self.media_processor.save_video_landmarks(all_frame_landmarks, landmark_filename)
            
            cap.release()
            if writer:
                writer.release()
                print(f"Webcam video saved to {video_filename}")
            if writer_black_background:
                writer_black_background.release()
                print(f"Webcam video with black background saved to {video_black_background_filename}")
            
            completion_message = "Webcam processing stopped."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.is_running = False

    def process_phone_stream(self, ip_address, plot_landmarks, plot_skeleton, plot_values, save_landmarks, save_video, landmark_filename, video_filename, save_video_black_background, video_black_background_filename):
        """A slot that processes a video stream from a phone camera app."""
        self.is_running = True
        writer = None
        writer_black_background = None

        try:
            url = f"http://{ip_address}:8080/shot.jpg"
            print(f"Attempting to connect to phone camera at: {url}")

            all_frame_landmarks = []
            
            # We need to get the first frame to initialize the video writer
            try:
                img_resp = requests.get(url, timeout=5)
                img_resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                first_frame = cv2.imdecode(img_arr, -1)
                
                if first_frame is None:
                    raise RuntimeError("Failed to decode the first frame from the phone camera. Check the IP Webcam app is running and the URL is correct.")
                
                if save_video and video_filename:
                    writer = self.init_writer_phone(video_filename, 7, first_frame)
                if save_video_black_background and video_black_background_filename:
                    writer_black_background = self.init_writer_phone(video_black_background_filename, 7, first_frame)

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Could not connect to phone camera. Check IP address and that the IP Webcam app is running. Error: {e}")

            while self.is_running:
                try:
                    img_resp = requests.get(url, timeout=1.5) # Use a timeout for the request
                    img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                
                    if frame is None:
                        print("Warning: Skipped a bad frame from phone camera.")
                        continue

                    # Reuse the same processing function as for videos/webcam
                    processed_frame, landmarks_dict, black_background_frame = self.media_processor.process_video_frame(frame, plot_landmarks, plot_skeleton, plot_values, save_video_black_background)

                    if landmarks_dict and save_landmarks:
                        all_frame_landmarks.append(landmarks_dict)

                    if save_video and video_filename:
                        self.new_frame_ready.emit(processed_frame)
                    elif save_video_black_background and video_black_background_filename and black_background_frame is not None:
                        self.new_frame_ready.emit(black_background_frame)
                    else:
                        self.new_frame_ready.emit(processed_frame)

                    if writer:
                        writer.write(processed_frame)
                    if writer_black_background:
                        writer_black_background.write(black_background_frame)

                    QCoreApplication.processEvents() # Keep UI responsive
                
                except requests.exceptions.RequestException:
                    # Don't stop the whole process, just log that a frame was missed.
                    print(f"Warning: Failed to get a frame from phone camera. Will retry.")
                    QCoreApplication.processEvents() 
                    continue # Try again on the next iteration

            # --- Post-loop cleanup ---
            if save_landmarks and landmark_filename:
                self.media_processor.save_video_landmarks(all_frame_landmarks, landmark_filename)
            
            completion_message = "Phone camera processing stopped."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if writer:
                writer.release()
                print(f"Phone video saved to {video_filename}")
            if writer_black_background:
                writer_black_background.release()
                print(f"Phone video with black background saved to {video_black_background_filename}")
            self.is_running = False

    def process_3d_video(self, video_path, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        
        self.is_running = True
        cap = None
        writer = None
        writer_black = None
        server = None
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            # Initialize video writers if needed
            ret, frame = cap.read()
            if save_video and video_filename:
                writer = self.init_writer(cap, video_filename, frame)
            if save_video_black and video_filename_black:
                writer_black = self.init_writer(cap, video_filename_black, frame)
            
            all_video_keypoints = []

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                display_frame = frame.copy()
                black_background_frame = np.zeros_like(frame) if save_video_black else None

                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_world_landmarks:
                    # Main 3D pipeline
                    landmarks_3d = extract_3D_landmarks(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d)
                    required_landmarks_3d = get_required_landmark(landmarks_3d, extra_landmarks_3d)
                                        
                    if save_keypoints_flag:
                        all_video_keypoints.append(required_landmarks_3d)

                    if server:
                        server.broadcast(required_landmarks_3d)

                    # Drawing pipeline (needs 2D landmarks)
                    if plot_landmarks_skeleton or plot_values:
                        landmarks_2d = extract_2D_landmarks(results)
                        extra_landmarks_2d = calculate_extra_landmarks(landmarks_2d)
                        required_landmarks_2d = get_required_landmark(landmarks_2d, extra_landmarks_2d)
                        denormalize_landmarks(display_frame, required_landmarks_2d)

                        if plot_landmarks_skeleton:
                            project_landmarks(display_frame, required_landmarks_2d)
                            project_skeleton(display_frame, required_landmarks_2d)
                            if save_video_black:
                                project_landmarks(black_background_frame, required_landmarks_2d)
                                project_skeleton(black_background_frame, required_landmarks_2d)
                        
                        if plot_values:
                            project_special_values(display_frame, required_landmarks_2d, required_landmarks_3d)
                            if save_video_black:
                                project_special_values(black_background_frame, required_landmarks_2d, required_landmarks_3d)
                
                if save_video:
                    self.new_frame_ready.emit(display_frame)
                elif save_video_black and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(display_frame)
                
                # Write frames to video files
                if writer:
                    writer.write(display_frame)
                if writer_black:
                    writer_black.write(black_background_frame)

                QCoreApplication.processEvents()
            
            # --- Cleanup ---
            if save_keypoints_flag and keypoints_filename:
                save_keypoints(all_video_keypoints, keypoints_filename)
            
            completion_message = "3D processing complete." if self.is_running else "Processing stopped by user."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            if writer_black:
                writer_black.release()
            if server:
                server.stop()
            self.is_running = False

    def process_3d_webcam(self, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        self.is_running = True
        cap = None
        writer = None
        writer_black = None
        server = None
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam.")
            
            fps = 15 # Assume a reasonable FPS for webcam saving

            # --- Main Processing ---
            if save_video:
                writer = self.init_writer_webcam(cap, video_filename, fps)
            if save_video_black:
                writer_black = self.init_writer_webcam(cap, video_filename_black, fps)
            
            all_video_keypoints = []

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("Failed to get frame from webcam.")
                    break

                display_frame = frame.copy()
                black_background_frame = np.zeros_like(frame) if save_video_black else None

                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_world_landmarks:
                    landmarks_3d = extract_3D_landmarks(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d)
                    required_landmarks_3d = get_required_landmark(landmarks_3d, extra_landmarks_3d)
                    
                    if save_keypoints_flag:
                        all_video_keypoints.append(required_landmarks_3d)
                    
                    if server:
                        server.broadcast(required_landmarks_3d)

                    if plot_landmarks_skeleton or plot_values:
                        landmarks_2d = extract_2D_landmarks(results)
                        extra_landmarks_2d = calculate_extra_landmarks(landmarks_2d)
                        required_landmarks_2d = get_required_landmark(landmarks_2d, extra_landmarks_2d)
                        denormalize_landmarks(display_frame, required_landmarks_2d)

                        if plot_landmarks_skeleton:
                            project_landmarks(display_frame, required_landmarks_2d)
                            project_skeleton(display_frame, required_landmarks_2d)
                            if save_video_black:
                                project_landmarks(black_background_frame, required_landmarks_2d)
                                project_skeleton(black_background_frame, required_landmarks_2d)
                        
                        if plot_values:
                            project_special_values(display_frame, required_landmarks_2d, required_landmarks_3d)
                            if save_video_black:
                                project_special_values(black_background_frame, required_landmarks_2d, required_landmarks_3d)
                
                if save_video:
                    self.new_frame_ready.emit(display_frame)
                elif save_video_black and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(display_frame)
                
                if writer:
                    writer.write(display_frame)
                if writer_black and black_background_frame is not None:
                    writer_black.write(black_background_frame)

                QCoreApplication.processEvents()
            
            if save_keypoints_flag and keypoints_filename:
                save_keypoints(all_video_keypoints, keypoints_filename)
            
            completion_message = "3D Webcam processing complete." if self.is_running else "Processing stopped by user."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            if writer_black:
                writer_black.release()
            if server:
                server.stop()
            self.is_running = False

    def process_3d_phone(self, ip_address, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        self.is_running = True
        writer = None
        writer_black = None
        server = None
        
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()
            
            url = f"http://{ip_address}:8080/shot.jpg"
            print(f"Attempting to connect to phone camera at: {url}")
            
            # Try to get one frame to initialize writers
            first_frame = None
            try:
                img_resp = requests.get(url, timeout=5)
                img_resp.raise_for_status()
                img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                first_frame = cv2.imdecode(img_arr, -1)
                if first_frame is None:
                    raise RuntimeError("Failed to decode frame for writer initialization.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Could not connect to phone camera to initialize. Check IP. Error: {e}")

            # --- Main Processing ---
            if save_video:
                writer = self.init_writer_phone(video_filename, 7, first_frame)
            if save_video_black:
                writer_black = self.init_writer_phone(video_filename_black, 7, first_frame)
            
            all_video_keypoints = []

            while self.is_running:
                try:
                    img_resp = requests.get(url, timeout=1.5)
                    img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                    if frame is None:
                        print("Warning: Skipped a bad frame from phone camera.")
                        continue
                except requests.exceptions.RequestException:
                    print(f"Warning: Failed to get a frame from phone camera. Will retry.")
                    QCoreApplication.processEvents()
                    continue

                display_frame = frame.copy()
                black_background_frame = np.zeros_like(frame) if save_video_black else None

                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_world_landmarks:
                    landmarks_3d = extract_3D_landmarks(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d)
                    required_landmarks_3d = get_required_landmark(landmarks_3d, extra_landmarks_3d)

                    if save_keypoints_flag:
                        all_video_keypoints.append(required_landmarks_3d)
                    
                    if server:
                        server.broadcast(required_landmarks_3d)

                    if plot_landmarks_skeleton or plot_values:
                        landmarks_2d = extract_2D_landmarks(results)
                        extra_landmarks_2d = calculate_extra_landmarks(landmarks_2d)
                        required_landmarks_2d = get_required_landmark(landmarks_2d, extra_landmarks_2d)
                        denormalize_landmarks(display_frame, required_landmarks_2d)

                        if plot_landmarks_skeleton:
                            project_landmarks(display_frame, required_landmarks_2d)
                            project_skeleton(display_frame, required_landmarks_2d)
                            if save_video_black and black_background_frame is not None:
                                project_landmarks(black_background_frame, required_landmarks_2d)
                                project_skeleton(black_background_frame, required_landmarks_2d)
                        
                        if plot_values:
                            project_special_values(display_frame, required_landmarks_2d, required_landmarks_3d)
                            if save_video_black and black_background_frame is not None:
                                project_special_values(black_background_frame, required_landmarks_2d, required_landmarks_3d)
                
                if save_video:
                    self.new_frame_ready.emit(display_frame)
                elif save_video_black and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(display_frame)
                
                if writer:
                    writer.write(display_frame)
                if writer_black and black_background_frame is not None:
                    writer_black.write(black_background_frame)

                QCoreApplication.processEvents()
            
            if save_keypoints_flag and keypoints_filename:
                save_keypoints(all_video_keypoints, keypoints_filename)
            
            completion_message = "3D Phone processing complete." if self.is_running else "Processing stopped by user."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if writer:
                writer.release()
            if writer_black:
                writer_black.release()
            if server:
                server.stop()
            self.is_running = False

    def process_3d_phone_with_depth_model(self, ip_address, use_depth_model, display_depth_map, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        self.is_running = True
        writer = None
        writer_black = None
        server = None
        is_first = True
        
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()
            
            url = f"http://{ip_address}:8080/shot.jpg"
            print(f"Attempting to connect to phone camera at: {url}")
            
            # Try to get one frame to initialize writers
            first_frame = None
            try:
                img_resp = requests.get(url, timeout=5)
                img_resp.raise_for_status()
                img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                first_frame = cv2.imdecode(img_arr, -1)
                if first_frame is None:
                    raise RuntimeError("Failed to decode frame for writer initialization.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Could not connect to phone camera to initialize. Check IP. Error: {e}")

            # --- Main Processing ---
            if save_video:
                writer = self.init_writer_phone(video_filename, 7, first_frame)
            if save_video_black:
                writer_black = self.init_writer_phone(video_filename_black, 7, first_frame)
            
            all_video_keypoints = []
            store_last_10_frames = []
            colored_map = None
            
            while self.is_running:
                try:
                    img_resp = requests.get(url, timeout=1.5)
                    img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                    if frame is None:
                        print("Warning: Skipped a bad frame from phone camera.")
                        continue
                except requests.exceptions.RequestException:
                    print(f"Warning: Failed to get a frame from phone camera. Will retry.")
                    QCoreApplication.processEvents()
                    continue

                display_frame = frame.copy()
                black_background_frame = np.zeros_like(frame) if save_video_black else None

                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_world_landmarks:
                    landmarks_3d = extract_3D_landmarks(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d)
                    required_landmarks_3d = get_required_landmark(landmarks_3d, extra_landmarks_3d)

                    if use_depth_model:
                        # Use the Depth model and get the depth value for the hip keypoint
                        with torch.no_grad():  # Disable gradient computation for faster inference
                            depth_map = self.model.infer_image(display_frame)
                            
                        # Get the Z value from the depth map and store it in a list
                        hip_z = float(get_depth_for_hip_keypoint(required_landmarks_3d, depth_map, display_frame))
                        store_last_10_frames.append(hip_z)
                        
                        if is_first:
                            first_z = hip_z
                            is_first = False
                        
                        # check if you have more then 5 fram pass to start shifting the keypoints
                        if len(store_last_10_frames) > 10:
                            length = len(store_last_10_frames)
                            hip_z_avg = float(round(np.mean(store_last_10_frames[length-10:]), 3))
                            shifting_keypoints_with_z_value(required_landmarks_3d, hip_z_avg, first_z)
                        else:
                            shifting_keypoints_with_z_value(required_landmarks_3d, hip_z, first_z)
                        
                        if display_depth_map:
                            colored_map = self.process_depth_map(depth_map)
                        
                    norm_hip_x = get_norm_x_for_hip(results)
                    shifting_keypoints_with_x_value(norm_hip_x, display_frame, required_landmarks_3d)
                    

                    if save_keypoints_flag:
                        all_video_keypoints.append(required_landmarks_3d)
                    
                    if server:
                        server.broadcast(required_landmarks_3d)
                    
                    if plot_landmarks_skeleton or plot_values:
                        landmarks_2d = extract_2D_landmarks(results)
                        extra_landmarks_2d = calculate_extra_landmarks(landmarks_2d)
                        required_landmarks_2d = get_required_landmark(landmarks_2d, extra_landmarks_2d)
                        denormalize_landmarks(display_frame, required_landmarks_2d)

                        if plot_landmarks_skeleton:
                            project_landmarks(display_frame, required_landmarks_2d)
                            project_skeleton(display_frame, required_landmarks_2d)
                            if save_video_black and black_background_frame is not None:
                                project_landmarks(black_background_frame, required_landmarks_2d)
                                project_skeleton(black_background_frame, required_landmarks_2d)
                        
                        if plot_values:
                            project_special_values(display_frame, required_landmarks_2d, required_landmarks_3d)
                            if save_video_black and black_background_frame is not None:
                                project_special_values(black_background_frame, required_landmarks_2d, required_landmarks_3d)
                
                if display_depth_map and colored_map is not None:
                    self.new_frame_ready.emit(colored_map)
                elif save_video:
                    self.new_frame_ready.emit(display_frame)
                elif save_video_black and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(display_frame)
                
                if writer:
                    writer.write(display_frame)
                if writer_black and black_background_frame is not None:
                    writer_black.write(black_background_frame)

                QCoreApplication.processEvents()
            
            if save_keypoints_flag and keypoints_filename:
                save_keypoints(all_video_keypoints, keypoints_filename)
            
            completion_message = "3D Phone processing complete." if self.is_running else "Processing stopped by user."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if writer:
                writer.release()
            if writer_black:
                writer_black.release()
            if server:
                server.stop()
            self.is_running = False

    def process_3d_video_with_depth_model(self, video_path, use_depth_model, display_depth_map, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        
        self.is_running = True
        cap = None
        writer = None
        writer_black = None
        server = None
        is_first = True
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            # Initialize video writers if needed
            ret, frame = cap.read()
            if save_video and video_filename:
                writer = self.init_writer(cap, video_filename, frame)
            if save_video_black and video_filename_black:
                writer_black = self.init_writer(cap, video_filename_black, frame)
            
            all_video_keypoints = []
            store_last_10_frames = []
            colored_map = None
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                display_frame = frame.copy()
                black_background_frame = np.zeros_like(frame) if save_video_black else None

                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_world_landmarks:
                    # Main 3D pipeline
                    landmarks_3d = extract_3D_landmarks(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d)
                    required_landmarks_3d = get_required_landmark(landmarks_3d, extra_landmarks_3d)
                    
                    if use_depth_model:
                        # Use the Depth model and get the depth value for the hip keypoint
                        with torch.no_grad():  # Disable gradient computation for faster inference
                            depth_map = self.model.infer_image(display_frame)
                            
                        # Get the Z value from the depth map and store it in a list
                        hip_z = float(get_depth_for_hip_keypoint(required_landmarks_3d, depth_map, display_frame))
                        store_last_10_frames.append(hip_z)
                        
                        if is_first:
                            first_z = hip_z
                            is_first = False
                        
                        # check if you have more then 5 fram pass to start shifting the keypoints
                        if len(store_last_10_frames) > 10:
                            length = len(store_last_10_frames)
                            hip_z_avg = float(round(np.mean(store_last_10_frames[length-10:]), 3))
                            shifting_keypoints_with_z_value(required_landmarks_3d, hip_z_avg, first_z)
                        else:
                            shifting_keypoints_with_z_value(required_landmarks_3d, hip_z, first_z)
                        
                        if display_depth_map:
                            colored_map = self.process_depth_map(depth_map)
                        
                    norm_hip_x = get_norm_x_for_hip(results)
                    shifting_keypoints_with_x_value(norm_hip_x, display_frame, required_landmarks_3d)
                    
                    if save_keypoints_flag:
                        all_video_keypoints.append(required_landmarks_3d)

                    if server:
                        server.broadcast(required_landmarks_3d)

                    # Drawing pipeline (needs 2D landmarks)
                    if plot_landmarks_skeleton or plot_values:
                        landmarks_2d = extract_2D_landmarks(results)
                        extra_landmarks_2d = calculate_extra_landmarks(landmarks_2d)
                        required_landmarks_2d = get_required_landmark(landmarks_2d, extra_landmarks_2d)
                        denormalize_landmarks(display_frame, required_landmarks_2d)

                        if plot_landmarks_skeleton:
                            project_landmarks(display_frame, required_landmarks_2d)
                            project_skeleton(display_frame, required_landmarks_2d)
                            if save_video_black:
                                project_landmarks(black_background_frame, required_landmarks_2d)
                                project_skeleton(black_background_frame, required_landmarks_2d)
                        
                        if plot_values:
                            project_special_values(display_frame, required_landmarks_2d, required_landmarks_3d)
                            if save_video_black:
                                project_special_values(black_background_frame, required_landmarks_2d, required_landmarks_3d)
                
                if display_depth_map and colored_map is not None:
                    self.new_frame_ready.emit(colored_map)
                elif save_video:
                    self.new_frame_ready.emit(display_frame)
                elif save_video_black and black_background_frame is not None:
                    self.new_frame_ready.emit(black_background_frame)
                else:
                    self.new_frame_ready.emit(display_frame)
                
                # Write frames to video files
                if writer:
                    writer.write(display_frame)
                if writer_black:
                    writer_black.write(black_background_frame)

                QCoreApplication.processEvents()
            
            # --- Cleanup ---
            if save_keypoints_flag and keypoints_filename:
                save_keypoints(all_video_keypoints, keypoints_filename)
            
            completion_message = "3D processing complete." if self.is_running else "Processing stopped by user."
            self.video_finished.emit(completion_message)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            if writer_black:
                writer_black.release()
            if server:
                server.stop()
            self.is_running = False

    def stop(self):
        """A slot to stop the processing loop."""
        print("Worker received stop signal.")
        self.is_running = False

    def close(self):
        """A slot to clean up the media processor."""
        self.stop() # Ensure processing is stopped before closing
        self.media_processor.close() 
        
    def init_writer(self, cap, video_filename, frame):
        video_path = os.path.join('outputs', 'videos', video_filename)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        return writer
    
    def init_writer_webcam(self, cap, video_filename, fps):
        video_path = os.path.join('outputs', 'videos', video_filename)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        return writer
    
    def init_writer_phone(self, video_filename, fps, frame):
        video_path = os.path.join('outputs', 'videos', video_filename)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        return writer

    def process_depth_map(self, depth):
        """
        Process raw depth map to colored visualization
        
        Args:
            depth (numpy.ndarray): Raw depth map
            
        Returns:
            tuple: (colored_depth_map, raw_depth_map) - BGR format for OpenCV
        """
        # Normalize depth values to 0-255 range for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap (convert from RGB to BGR for OpenCV)
        colored_depth = (self.cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        return colored_depth