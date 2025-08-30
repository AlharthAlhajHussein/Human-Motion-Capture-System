from PyQt6.QtCore import QObject, pyqtSignal, QCoreApplication
from logic.media_processor import MediaProcessor
import cv2
import os
import requests
import numpy as np
import time
from logic.system_functions import (
    extract_2D_landmarks,
    extract_3D_landmarks_fixed,
    calculate_extra_landmarks,
    get_required_landmark,
    calculate_scaling_params,
    scale_landmarks,
    denormalize_landmarks,
    project_landmarks,
    project_skeleton,
    project_special_values,
    save_keypoints,
)
from logic.websocket_server import KeypointServer


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
    status_update = pyqtSignal(str)     # The 'str' will be a status message for the UI

    def __init__(self):
        super().__init__()
        # The MediaProcessor is now owned by the worker
        self.media_processor = MediaProcessor()
        self.is_running = False # Flag to control the processing loop

    def switch_model(self, model_comp):
        """Switches the model complexity
        Args:
            model_complixity (int): The complexity of the model 1 light or 2 heavy
        """
        self.media_processor = MediaProcessor(model_complexity=model_comp)
    
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

    def process_3d_video(self, video_path, scaling_time, message_time, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        self.is_running = True
        cap = None
        writer = None
        writer_black = None
        server = None
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()
            # --- Phase 1: Calibration ---
            for i in range(message_time):
                self.status_update.emit(f"Prepare for T-Pose calibration in {message_time-i} seconds...")
                time.sleep(1)
            

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            calibration_frames_count = int(scaling_time * fps)
            all_calibration_landmarks = []
            
            for i in range(calibration_frames_count):
                if not self.is_running:
                    self.video_finished.emit("Processing stopped by user during calibration.")
                    return
                
                ret, frame = cap.read()
                if not ret:
                    break # End of video before calibration finished

                # Display countdown on UI
                remaining_time = scaling_time - (i / fps)
                self.status_update.emit(f"Hold T-Pose... Calibrating: {remaining_time:.1f}s remaining")
                self.new_frame_ready.emit(frame) # Show the raw frame during calibration
                
                # Process for landmarks
                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_world_landmarks:
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks = get_required_landmark(landmarks_3d_fixed, extra_landmarks)
                    all_calibration_landmarks.append(required_landmarks)
                
                QCoreApplication.processEvents()

            if not all_calibration_landmarks:
                raise RuntimeError("Could not detect any landmarks during the calibration phase. Please ensure you are visible and in T-pose.")
            
            # Average the landmarks from the calibration phase
            avg_landmarks = {}
            for key in all_calibration_landmarks[0].keys():
                avg_landmarks[key] = {
                    'x': np.mean([frame[key]['x'] for frame in all_calibration_landmarks]),
                    'y': np.mean([frame[key]['y'] for frame in all_calibration_landmarks]),
                    'z': np.mean([frame[key]['z'] for frame in all_calibration_landmarks]),
                    'name': key
                }

            # Calculate scaling parameters
            w1, w2, lowest_point = calculate_scaling_params(avg_landmarks)
            self.status_update.emit("Calibration Complete. Starting main processing...")
            time.sleep(2) # Give user time to see the message

            # --- Phase 2: Main Processing ---
            # Reset video capture to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
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
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks_3d = get_required_landmark(landmarks_3d_fixed, extra_landmarks_3d)
                    
                    # Create a copy for scaling, so the original isn't modified if needed elsewhere
                    scaled_landmarks = {k: v.copy() for k, v in required_landmarks_3d.items()}
                    scale_landmarks(scaled_landmarks, w1, w2, lowest_point)

                    if save_keypoints_flag:
                        all_video_keypoints.append(scaled_landmarks)

                    if server:
                        server.broadcast(scaled_landmarks)

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
                            project_special_values(display_frame, required_landmarks_2d, scaled_landmarks)
                            if save_video_black:
                                project_special_values(black_background_frame, required_landmarks_2d, scaled_landmarks)
                
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
            
            completion_message = "3D Fixed processing complete." if self.is_running else "Processing stopped by user."
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

    def process_3d_webcam(self, scaling_time, message_time, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
        self.is_running = True
        cap = None
        writer = None
        writer_black = None
        server = None
        try:
            if send_keypoints:
                server = KeypointServer(port)
                server.start()
            
            # --- Phase 1: Calibration ---
            for i in range(message_time):
                self.status_update.emit(f"Prepare for T-Pose calibration in {message_time-i} seconds...")
                time.sleep(1)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam.")
            
            fps = 15 # Assume a reasonable FPS for webcam saving
            start_time = time.time()
            all_calibration_landmarks = []
            
            while time.time() - start_time < scaling_time:
                if not self.is_running:
                    self.video_finished.emit("Processing stopped by user during calibration.")
                    return
                
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("Failed to get frame from webcam during calibration.")
                    break

                remaining_time = scaling_time - (time.time() - start_time)
                self.status_update.emit(f"Hold T-Pose... Calibrating: {remaining_time:.1f}s remaining")
                self.new_frame_ready.emit(frame)
                
                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_world_landmarks:
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks = get_required_landmark(landmarks_3d_fixed, extra_landmarks)
                    all_calibration_landmarks.append(required_landmarks)
                
                QCoreApplication.processEvents()

            if not all_calibration_landmarks:
                raise RuntimeError("Could not detect any landmarks during the calibration phase. Please ensure you are visible and in T-pose.")
            
            avg_landmarks = {}
            for key in all_calibration_landmarks[0].keys():
                avg_landmarks[key] = {
                    'x': np.mean([frame[key]['x'] for frame in all_calibration_landmarks]),
                    'y': np.mean([frame[key]['y'] for frame in all_calibration_landmarks]),
                    'z': np.mean([frame[key]['z'] for frame in all_calibration_landmarks]),
                    'name': key
                }

            w1, w2, lowest_point = calculate_scaling_params(avg_landmarks)
            self.status_update.emit("Calibration Complete. Starting main processing...")
            time.sleep(2)

            # --- Phase 2: Main Processing ---
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
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks_3d = get_required_landmark(landmarks_3d_fixed, extra_landmarks_3d)
                    
                    scaled_landmarks = {k: v.copy() for k, v in required_landmarks_3d.items()}
                    scale_landmarks(scaled_landmarks, w1, w2, lowest_point)

                    if save_keypoints_flag:
                        all_video_keypoints.append(scaled_landmarks)
                    
                    if server:
                        server.broadcast(scaled_landmarks)

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
                            project_special_values(display_frame, required_landmarks_2d, scaled_landmarks)
                            if save_video_black:
                                project_special_values(black_background_frame, required_landmarks_2d, scaled_landmarks)
                
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
            
            completion_message = "3D Fixed Webcam processing complete." if self.is_running else "Processing stopped by user."
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

    def process_3d_phone(self, ip_address, scaling_time, message_time, plot_landmarks_skeleton, plot_values, save_keypoints_flag, keypoints_filename, save_video, video_filename, save_video_black, video_filename_black, send_keypoints, port):
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

            # --- Phase 1: Calibration ---
            for i in range(message_time):
                self.status_update.emit(f"Prepare for T-Pose calibration in {message_time-i} seconds...")
                time.sleep(1)
            
            start_time = time.time()
            all_calibration_landmarks = []
            
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

            while time.time() - start_time < scaling_time:
                if not self.is_running:
                    self.video_finished.emit("Processing stopped by user during calibration.")
                    return
                
                try:
                    img_resp = requests.get(url, timeout=1.5)
                    img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                    if frame is None:
                        print("Warning: Skipped a bad frame from phone during calibration.")
                        continue
                except requests.exceptions.RequestException:
                    print(f"Warning: Failed to get a frame from phone camera. Will retry.")
                    QCoreApplication.processEvents()
                    continue

                remaining_time = scaling_time - (time.time() - start_time)
                self.status_update.emit(f"Hold T-Pose... Calibrating: {remaining_time:.1f}s remaining")
                self.new_frame_ready.emit(frame)
                
                results = self.media_processor.video_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_world_landmarks:
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks = get_required_landmark(landmarks_3d_fixed, extra_landmarks)
                    all_calibration_landmarks.append(required_landmarks)
                
                QCoreApplication.processEvents()

            if not all_calibration_landmarks:
                raise RuntimeError("Could not detect any landmarks during the calibration phase. Please ensure you are visible and in T-pose.")
            
            avg_landmarks = {}
            for key in all_calibration_landmarks[0].keys():
                avg_landmarks[key] = {
                    'x': np.mean([frame[key]['x'] for frame in all_calibration_landmarks]),
                    'y': np.mean([frame[key]['y'] for frame in all_calibration_landmarks]),
                    'z': np.mean([frame[key]['z'] for frame in all_calibration_landmarks]),
                    'name': key
                }

            w1, w2, lowest_point = calculate_scaling_params(avg_landmarks)
            self.status_update.emit("Calibration Complete. Starting main processing...")
            time.sleep(2)

            # --- Phase 2: Main Processing ---
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
                    landmarks_3d_fixed = extract_3D_landmarks_fixed(results)
                    extra_landmarks_3d = calculate_extra_landmarks(landmarks_3d_fixed)
                    required_landmarks_3d = get_required_landmark(landmarks_3d_fixed, extra_landmarks_3d)
                    
                    scaled_landmarks = {k: v.copy() for k, v in required_landmarks_3d.items()}
                    scale_landmarks(scaled_landmarks, w1, w2, lowest_point)

                    if save_keypoints_flag:
                        all_video_keypoints.append(scaled_landmarks)
                    
                    if server:
                        server.broadcast(scaled_landmarks)

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
                            project_special_values(display_frame, required_landmarks_2d, scaled_landmarks)
                            if save_video_black and black_background_frame is not None:
                                project_special_values(black_background_frame, required_landmarks_2d, scaled_landmarks)
                
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
            
            completion_message = "3D Fixed Phone processing complete." if self.is_running else "Processing stopped by user."
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