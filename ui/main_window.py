import sys
from PyQt6.QtWidgets import (QApplication,
                             QWidget,
                             QPushButton,
                             QComboBox,
                             QVBoxLayout,
                             QLabel, 
                             QFileDialog, 
                             QCheckBox, 
                             QLineEdit, 
                             QHBoxLayout, 
                             QSpacerItem, 
                             QSizePolicy, 
                             QFrame,
                             QMessageBox,
                             QMenuBar,
                             QDialog,
                             QTextEdit)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import (QPixmap, 
                         QImage, 
                         QIcon,
                         QFont)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from logic.worker import Worker
from logic.system_functions import load_image_with_orientation
from .styles import *


class MainWindow(QWidget):
    # --- New Signals to trigger worker ---
    # We define signals here that will carry the data to the worker thread
    start_image_processing_signal = pyqtSignal(str, bool, bool, bool, str, bool, str, str, bool, str)
    start_video_processing_signal = pyqtSignal(str, bool, bool, bool, bool, bool, str, str, bool, str)
    start_webcam_processing_signal = pyqtSignal(bool, bool, bool, bool, bool, str, str, bool, str)
    start_phone_processing_signal = pyqtSignal(str, bool, bool, bool, bool, bool, str, str, bool, str)
    # 3D signals
    start_3d_video_processing_signal = pyqtSignal(str, bool, bool, bool, str, bool, str, bool, str, bool, int)
    start_3d_webcam_processing_signal = pyqtSignal(bool, bool, bool, str, bool, str, bool, str, bool, int)
    start_3d_phone_processing_signal = pyqtSignal(str, bool, bool, bool, str, bool, str, bool, str, bool, int)
    start_3d_depth_video_processing_signal = pyqtSignal(str, bool, bool, bool, bool, bool, str, bool, str, bool, str, bool, int)
    stop_worker_signal = pyqtSignal() 
    switch_the_model_signal = pyqtSignal(int)
    
    def __init__(self):
        # --- Initialize the superclass ---
        super().__init__()
        
        # --- Initialize the uploaded media paths ---
        self.uploaded_image_path = None
        self.uploaded_video_path = None
        
        self.initUI()
        self.setup_worker_thread() # New method to set up the thread
        self.apply_theme('light')  # Apply modern theme

    def initUI(self):
        self.setWindowTitle('Human Motion Capture System')
        self.setWindowIcon(QIcon('resources/icons/main_window_icon.png'))

        
        #region --- Create Menu Bar --- 
        self.menu_bar = QMenuBar(self)
        
        # --- Settings Menu ---
        # Theme settings
        settings_menu = self.menu_bar.addMenu('Settings')
        theme_menu = settings_menu.addMenu('Theme')
        light_mode_action = QAction('Light Mode', self)
        light_mode_action.triggered.connect(self.set_light_mode)
        theme_menu.addAction(light_mode_action)
        dark_mode_action = QAction('Dark Mode', self)
        dark_mode_action.triggered.connect(self.set_dark_mode)
        theme_menu.addAction(dark_mode_action)
        
        # Model settings
        model_menu = settings_menu.addMenu('Select Model')
        light_model_action = QAction('Light Model', self)
        light_model_action.triggered.connect(self.set_light_model)
        model_menu.addAction(light_model_action)
        heavy_model_action = QAction('Heavy Model', self)
        heavy_model_action.triggered.connect(self.set_heavy_model)
        model_menu.addAction(heavy_model_action)

        # --- Help Menu ---
        help_menu = self.menu_bar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        contact_action = QAction('Contact Us', self)
        contact_action.triggered.connect(self.show_contact_dialog)
        help_menu.addAction(contact_action)
        #endregion
        
        # --- Create the start and end buttons ---
        self.start_button = QPushButton('Start')
        self.start_button.setMinimumWidth(200)
        
        self.end_button = QPushButton('End')
        self.end_button.setMinimumWidth(200)
        
        # --- label to display the media ---
        self.displayed_media_label = QLabel('Your selected media will be displayed here.')
        self.displayed_media_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.displayed_media_label.setFrameStyle(QFrame.Shape.StyledPanel) 
        self.displayed_media_label.setMinimumSize(640, 480)
        
        # --- label for status updates ---
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: red;")
    
        # --- primary dropdown for selecting the type of processing ---
        self.dropdown = QComboBox()
        self.dropdown.addItems(['Select Type...', '2D', '3D'])
        
        
        # --- secondary 2D dropdown for selecting the source of the media ---
        self.dropdown_2d = QComboBox()
        self.dropdown_2d.addItems(['Select Source...', 'Upload Image', 'Upload Video', 'Use Webcam', 'Use Smartphone Camera'])
        self.dropdown_2d.hide()

        # --- secondary 3D dropdown for selecting the source of the media ---
        self.dropdown_3d = QComboBox()
        self.dropdown_3d.addItems(['Select Source...', 'Upload Video', 'Use Webcam', 'Use Smartphone Camera', 'Upload Video with Depth Model'])
        self.dropdown_3d.hide()

        #! name convention is : (type ofsource)_(type of processing)_(variable name)_(UI name)
        #region UI elements for 2D video processing 
        # buttons
        self.vid_browse_button = QPushButton('Browse for Video...')
        # checkboxes
        self.vid_plot_values_cb = QCheckBox("Plot Values on Video")
        self.vid_plot_landmarks_cb = QCheckBox("Plot Landmarks on Video")
        self.vid_plot_skeleton_cb = QCheckBox("Plot Skeleton on Video")
        self.vid_save_landmarks_cb = QCheckBox("Save Video Keypoints to File")
        self.vid_save_processed_video_cb = QCheckBox("Save Processed Video")
        self.vid_save_processed_video_black_background_cb = QCheckBox("Save Processed Video with Black Background")
        # labels
        self.vid_landmarks_filename_label = QLabel("Keypoints Filename:")
        self.vid_output_video_filename_label = QLabel("Processed Video Filename:")
        self.vid_output_video_black_background_filename_label = QLabel("Processed Video with Black Background:")
        # inputs
        self.vid_landmarks_filename_input = QLineEdit("video_keypoints.json")
        self.vid_output_video_filename_input = QLineEdit("processed_video.mp4")
        self.vid_output_video_black_background_filename_input = QLineEdit("processed_video_black_background.mp4")
        #endregion
        
        #region UI elements for 2D image processing 
        # buttons
        self.img_browse_button = QPushButton('Browse for Image...')
        # checkboxes
        self.img_plot_landmarks_cb = QCheckBox("Plot Landmarks")
        self.img_plot_skeleton_cb = QCheckBox("Plot Skeleton")
        self.img_save_landmarks_cb = QCheckBox("Save Landmarks to File")
        self.img_save_processed_image_cb = QCheckBox("Save Processed Image")
        self.img_save_processed_image_black_background_cb = QCheckBox("Save Processed Image with Black Background")
        # labels
        self.img_landmarks_filename_label = QLabel("Keypoints Filename:")
        self.img_processed_image_filename_label = QLabel("Processed Image Filename:")
        self.img_processed_image_size_label = QLabel("Output Size:")
        self.img_processed_image_black_background_filename_label = QLabel("Processed Image with Black Background:")
        # inputs
        self.img_landmarks_filename_input = QLineEdit("image_landmarks.json")
        self.img_processed_image_filename_input = QLineEdit("processed_image.png")
        self.img_processed_image_black_background_filename_input = QLineEdit("processed_image_black_background.png")
        # dropdowns
        self.img_processed_image_size_combo = QComboBox()
        self.img_processed_image_size_combo.addItems(["Original", "1920x1080", "1280x720", "640x480"])
        #endregion

        #region UI elements for 2D webcam processing 
        # checkboxes
        self.cam_plot_values_cb = QCheckBox("Plot Values on Webcam Video")
        self.cam_plot_landmarks_cb = QCheckBox("Plot Landmarks on Webcam Video")
        self.cam_plot_skeleton_cb = QCheckBox("Plot Skeleton on Webcam Video")
        self.cam_save_landmarks_cb = QCheckBox("Save Webcam Keypoints to File")
        self.cam_save_processed_video_cb = QCheckBox("Save Processed Webcam Video")
        self.cam_save_processed_video_black_background_cb = QCheckBox("Save Processed Webcam Video with Black Background")
        # labels
        self.cam_landmarks_filename_label = QLabel("Keypoints Filename:")
        self.cam_output_video_filename_label = QLabel("Processed Video Filename:")
        self.cam_output_video_black_background_filename_label = QLabel("Processed Video with Black Background:")
        # inputs
        self.cam_landmarks_filename_input = QLineEdit("webcam_keypoints.json")
        self.cam_output_video_filename_input = QLineEdit("webcam_video.mp4")
        self.cam_output_video_black_background_filename_input = QLineEdit("webcam_video_black_background.mp4")
        #endregion

        #region UI elements for 2D phone camera processing 
        # labels
        self.phone_ip_label = QLabel("Phone IP Address:")
        self.phone_landmarks_filename_label = QLabel("Keypoints Filename:")
        self.phone_output_video_filename_label = QLabel("Processed Video Filename:")
        self.phone_output_video_black_background_filename_label = QLabel("Processed Video with Black Background:")
        # inputs
        self.phone_ip_input = QLineEdit("192.168.1.107") # Default, user should change
        self.phone_landmarks_filename_input = QLineEdit("phone_keypoints.json")
        self.phone_output_video_filename_input = QLineEdit("phone_video.mp4")
        self.phone_output_video_black_background_filename_input = QLineEdit("phone_video_black_background.mp4")
        # checkboxes
        self.phone_plot_values_cb = QCheckBox("Plot Values on Phone Video")
        self.phone_plot_landmarks_cb = QCheckBox("Plot Landmarks on Phone Video")
        self.phone_plot_skeleton_cb = QCheckBox("Plot Skeleton on Phone Video")
        self.phone_save_landmarks_cb = QCheckBox("Save Phone Keypoints to File")
        self.phone_save_processed_video_cb = QCheckBox("Save Processed Phone Video")
        self.phone_save_processed_video_black_background_cb = QCheckBox("Save Processed Phone Video with Black Background")
        #endregion
        
        #region UI elements for 3D video processing
        # buttons
        self.vid_3d_browse_button = QPushButton('Browse for Video...')
        # labels
        self.vid_3d_keypoints_filename_label = QLabel("Keypoints Filename:")
        self.vid_3d_output_video_filename_label = QLabel("Processed Video Filename:")
        self.vid_3d_output_video_black_background_filename_label = QLabel("Processed Video with Black Background Filename:")
        self.vid_3d_port_label = QLabel("Port:")
        # inputs
        self.vid_3d_keypoints_filename_input = QLineEdit("keypoints_3d.json")
        self.vid_3d_output_video_filename_input = QLineEdit("processed_video_3d.mp4")
        self.vid_3d_output_video_black_background_filename_input = QLineEdit("processed_video_3d_black.mp4")
        self.vid_3d_port_input = QLineEdit("8000")
        # checkboxes
        self.vid_3d_plot_landmarks_cb = QCheckBox("Plot Landmarks and Skeleton")
        self.vid_3d_plot_values_cb = QCheckBox("Plot Values")
        self.vid_3d_save_keypoints_cb = QCheckBox("Save Keypoints as JSON")
        self.vid_3d_save_video_cb = QCheckBox("Save Video")
        self.vid_3d_save_video_black_background_cb = QCheckBox("Save Video with Black Background")
        self.vid_3d_send_keypoints_cb = QCheckBox("Send Keypoints over Network")
        #endregion

        #region UI elements for 3D webcam processing
        # labels
        self.cam_3d_keypoints_filename_label = QLabel("Keypoints Filename:")
        self.cam_3d_output_video_filename_label = QLabel("Processed Video Filename:")
        self.cam_3d_output_video_black_background_filename_label = QLabel("Processed Video with Black Background Filename:")
        self.cam_3d_port_label = QLabel("Port:")
        # inputs
        self.cam_3d_keypoints_filename_input = QLineEdit("keypoints_3d_webcam.json")
        self.cam_3d_output_video_filename_input = QLineEdit("processed_video_3d_webcam.mp4")
        self.cam_3d_output_video_black_background_filename_input = QLineEdit("processed_video_3d_webcam_black.mp4")
        self.cam_3d_port_input = QLineEdit("8000")
        # checkboxes
        self.cam_3d_plot_landmarks_cb = QCheckBox("Plot Landmarks and Skeleton")
        self.cam_3d_plot_values_cb = QCheckBox("Plot Values")
        self.cam_3d_save_keypoints_cb = QCheckBox("Save Keypoints as JSON")
        self.cam_3d_save_video_cb = QCheckBox("Save Video")
        self.cam_3d_save_video_black_background_cb = QCheckBox("Save Video with Black Background")
        self.cam_3d_send_keypoints_cb = QCheckBox("Send Keypoints over Network")
        #endregion

        #region UI elements for 3D phone processing
        # labels
        self.phone_3d_ip_label = QLabel("Phone IP Address:")
        self.phone_3d_keypoints_filename_label = QLabel("Keypoints Filename:")
        self.phone_3d_output_video_filename_label = QLabel("Processed Video Filename:")
        self.phone_3d_output_video_black_background_filename_label = QLabel("Processed Video with Black Background Filename:")
        self.phone_3d_port_label = QLabel("Port:")
        # inputs
        self.phone_3d_ip_input = QLineEdit("192.168.1.107") # Default, user should change
        self.phone_3d_keypoints_filename_input = QLineEdit("keypoints_3d_phone.json")
        self.phone_3d_output_video_filename_input = QLineEdit("processed_video_3d_phone.mp4")
        self.phone_3d_output_video_black_background_filename_input = QLineEdit("processed_video_3d_phone_black.mp4")
        self.phone_3d_port_input = QLineEdit("8000")
        # checkboxes
        self.phone_3d_plot_landmarks_cb = QCheckBox("Plot Landmarks and Skeleton")
        self.phone_3d_plot_values_cb = QCheckBox("Plot Values")
        self.phone_3d_save_keypoints_cb = QCheckBox("Save Keypoints as JSON")
        self.phone_3d_save_video_cb = QCheckBox("Save Video")
        self.phone_3d_save_video_black_background_cb = QCheckBox("Save Video with Black Background")
        self.phone_3d_send_keypoints_cb = QCheckBox("Send Keypoints over Network")
        #endregion

        #region UI elements for 3D video processing with Depth anything V2 model
        # buttons
        self.vid_3d_depth_browse_button = QPushButton('Browse for Video...')
        # labels
        self.vid_3d_depth_keypoints_filename_label = QLabel("Keypoints Filename:")
        self.vid_3d_depth_output_video_filename_label = QLabel("Processed Video Filename:")
        self.vid_3d_depth_output_video_black_background_filename_label = QLabel("Processed Video with Black Background Filename:")
        self.vid_3d_depth_port_label = QLabel("Port:")
        # inputs
        self.vid_3d_depth_keypoints_filename_input = QLineEdit("keypoints_3d.json")
        self.vid_3d_depth_output_video_filename_input = QLineEdit("processed_video_3d.mp4")
        self.vid_3d_depth_output_video_black_background_filename_input = QLineEdit("processed_video_3d_black.mp4")
        self.vid_3d_depth_port_input = QLineEdit("8000")
        # checkboxes
        self.vid_3d_depth_disp_depth_map_ch = QCheckBox("Display The Depth Map")
        self.vid_3d_depth_use_depth_model_cb = QCheckBox("Use The Depth Anything v2 Model")
        self.vid_3d_depth_plot_landmarks_cb = QCheckBox("Plot Landmarks and Skeleton")
        self.vid_3d_depth_plot_values_cb = QCheckBox("Plot Values")
        self.vid_3d_depth_save_keypoints_cb = QCheckBox("Save Keypoints as JSON")
        self.vid_3d_depth_save_video_cb = QCheckBox("Save Video")
        self.vid_3d_depth_save_video_black_background_cb = QCheckBox("Save Video with Black Background")
        self.vid_3d_depth_send_keypoints_cb = QCheckBox("Send Keypoints over Network")
        #endregion
        
        # --- Hide all elements by default ---
        self.hide_all_2D_sources_options()
        self.hide_all_3D_sources_options()
        
        # --- Connect the UI elements to the appropriate functions ---
        self.dropdown.currentIndexChanged.connect(self.on_dimension_changed)
        self.dropdown_2d.currentIndexChanged.connect(self.on_source_2D_changed)
        self.dropdown_3d.currentIndexChanged.connect(self.on_source_3D_changed)
        self.start_button.clicked.connect(self.start_processing)
        self.end_button.clicked.connect(self.stop_processing)
        
        #region connect for 2D UI elements 
        self.img_browse_button.clicked.connect(self.open_image_dialog)
        self.vid_browse_button.clicked.connect(self.open_video_dialog)
        self.img_save_landmarks_cb.stateChanged.connect(self.on_image_save_landmarks_changed)
        self.img_save_processed_image_cb.stateChanged.connect(self.on_image_save_processed_image_changed)
        self.img_save_processed_image_black_background_cb.stateChanged.connect(self.on_image_save_processed_image_black_background_changed)
        self.vid_save_landmarks_cb.stateChanged.connect(self.on_video_save_landmarks_changed)
        self.vid_save_processed_video_cb.stateChanged.connect(self.on_video_save_processed_video_changed)
        self.vid_save_processed_video_black_background_cb.stateChanged.connect(self.on_video_save_processed_video_black_background_changed)
        self.cam_save_landmarks_cb.stateChanged.connect(self.on_webcam_save_landmarks_changed)
        self.cam_save_processed_video_cb.stateChanged.connect(self.on_webcam_save_processed_video_changed)
        self.cam_save_processed_video_black_background_cb.stateChanged.connect(self.on_webcam_save_processed_video_black_background_changed)
        self.phone_save_landmarks_cb.stateChanged.connect(self.on_phone_save_landmarks_changed)
        self.phone_save_processed_video_cb.stateChanged.connect(self.on_phone_save_processed_video_changed)
        self.phone_save_processed_video_black_background_cb.stateChanged.connect(self.on_phone_save_processed_video_black_background_changed)
        #endregion
        
        #region connect for 3D UI elements
        self.vid_3d_browse_button.clicked.connect(self.open_video_dialog) # Can reuse the same dialog
        self.vid_3d_save_keypoints_cb.stateChanged.connect(self.on_3d_video_save_keypoints_changed)
        self.vid_3d_save_video_cb.stateChanged.connect(self.on_3d_video_save_video_changed)
        self.vid_3d_save_video_black_background_cb.stateChanged.connect(self.on_3d_video_save_video_black_background_changed)
        self.vid_3d_send_keypoints_cb.stateChanged.connect(self.on_3d_video_send_keypoints_changed)
        
        self.cam_3d_save_keypoints_cb.stateChanged.connect(self.on_3d_cam_save_keypoints_changed)
        self.cam_3d_save_video_cb.stateChanged.connect(self.on_3d_cam_save_video_changed)
        self.cam_3d_save_video_black_background_cb.stateChanged.connect(self.on_3d_cam_save_video_black_background_changed)
        self.cam_3d_send_keypoints_cb.stateChanged.connect(self.on_3d_cam_send_keypoints_changed)

        self.phone_3d_save_keypoints_cb.stateChanged.connect(self.on_3d_phone_save_keypoints_changed)
        self.phone_3d_save_video_cb.stateChanged.connect(self.on_3d_phone_save_video_changed)
        self.phone_3d_save_video_black_background_cb.stateChanged.connect(self.on_3d_phone_save_video_black_background_changed)
        self.phone_3d_send_keypoints_cb.stateChanged.connect(self.on_3d_phone_send_keypoints_changed)
        
        self.vid_3d_depth_browse_button.clicked.connect(self.open_video_dialog) # Can reuse the same dialog
        self.vid_3d_depth_save_keypoints_cb.stateChanged.connect(self.on_3d_depth_video_save_keypoints_changed)
        self.vid_3d_depth_save_video_cb.stateChanged.connect(self.on_3d_depth_video_save_video_changed)
        self.vid_3d_depth_save_video_black_background_cb.stateChanged.connect(self.on_3d_depth_video_save_video_black_background_changed)
        self.vid_3d_depth_send_keypoints_cb.stateChanged.connect(self.on_3d_depth_video_send_keypoints_changed)
        
        #endregion

        # --- Main Layout ---
        # This is the main horizontal layout that will hold the two columns
        main_layout = QHBoxLayout()

        #* --- Right Column Layout for Controls ---
        #region add widgets to the controls layout
        # All controls will go into this vertical layout
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.end_button)
        controls_layout.addWidget(self.dropdown)
        controls_layout.addWidget(self.dropdown_2d)
        controls_layout.addWidget(self.dropdown_3d)
        # Add image widgets to layout
        controls_layout.addWidget(self.img_browse_button)
        controls_layout.addWidget(self.vid_browse_button)
        controls_layout.addWidget(self.img_plot_landmarks_cb)
        controls_layout.addWidget(self.img_plot_skeleton_cb)
        controls_layout.addWidget(self.img_save_landmarks_cb)
        controls_layout.addWidget(self.img_landmarks_filename_label)
        controls_layout.addWidget(self.img_landmarks_filename_input)
        controls_layout.addWidget(self.img_save_processed_image_cb)
        controls_layout.addWidget(self.img_processed_image_filename_label)
        controls_layout.addWidget(self.img_processed_image_filename_input)
        controls_layout.addWidget(self.img_save_processed_image_black_background_cb)
        controls_layout.addWidget(self.img_processed_image_black_background_filename_label)
        controls_layout.addWidget(self.img_processed_image_black_background_filename_input)
        controls_layout.addWidget(self.img_processed_image_size_label)
        controls_layout.addWidget(self.img_processed_image_size_combo)

        # Add video widgets to layout
        controls_layout.addWidget(self.vid_plot_landmarks_cb)
        controls_layout.addWidget(self.vid_plot_skeleton_cb)
        controls_layout.addWidget(self.vid_plot_values_cb)
        controls_layout.addWidget(self.vid_save_landmarks_cb)
        controls_layout.addWidget(self.vid_landmarks_filename_label)
        controls_layout.addWidget(self.vid_landmarks_filename_input)
        controls_layout.addWidget(self.vid_save_processed_video_cb)
        controls_layout.addWidget(self.vid_output_video_filename_label)
        controls_layout.addWidget(self.vid_output_video_filename_input)
        controls_layout.addWidget(self.vid_save_processed_video_black_background_cb)
        controls_layout.addWidget(self.vid_output_video_black_background_filename_label)
        controls_layout.addWidget(self.vid_output_video_black_background_filename_input)
        # Add webcam widgets to layout
        controls_layout.addWidget(self.cam_plot_landmarks_cb)
        controls_layout.addWidget(self.cam_plot_skeleton_cb)
        controls_layout.addWidget(self.cam_plot_values_cb)
        controls_layout.addWidget(self.cam_save_landmarks_cb)
        controls_layout.addWidget(self.cam_landmarks_filename_label)
        controls_layout.addWidget(self.cam_landmarks_filename_input)
        controls_layout.addWidget(self.cam_save_processed_video_cb)
        controls_layout.addWidget(self.cam_output_video_filename_label)
        controls_layout.addWidget(self.cam_output_video_filename_input)
        controls_layout.addWidget(self.cam_save_processed_video_black_background_cb)
        controls_layout.addWidget(self.cam_output_video_black_background_filename_label)
        controls_layout.addWidget(self.cam_output_video_black_background_filename_input)
        # Add phone widgets to layout
        controls_layout.addWidget(self.phone_ip_label)
        controls_layout.addWidget(self.phone_ip_input)
        controls_layout.addWidget(self.phone_plot_landmarks_cb)
        controls_layout.addWidget(self.phone_plot_skeleton_cb)
        controls_layout.addWidget(self.phone_plot_values_cb)
        controls_layout.addWidget(self.phone_save_landmarks_cb)
        controls_layout.addWidget(self.phone_landmarks_filename_label)
        controls_layout.addWidget(self.phone_landmarks_filename_input)
        controls_layout.addWidget(self.phone_save_processed_video_cb)
        controls_layout.addWidget(self.phone_output_video_filename_label)
        controls_layout.addWidget(self.phone_output_video_filename_input)
        controls_layout.addWidget(self.phone_save_processed_video_black_background_cb)
        controls_layout.addWidget(self.phone_output_video_black_background_filename_label)
        controls_layout.addWidget(self.phone_output_video_black_background_filename_input)
        # Add 3D video widgets to layout
        controls_layout.addWidget(self.vid_3d_browse_button)
        controls_layout.addWidget(self.vid_3d_plot_landmarks_cb)
        controls_layout.addWidget(self.vid_3d_plot_values_cb)
        controls_layout.addWidget(self.vid_3d_save_keypoints_cb)
        controls_layout.addWidget(self.vid_3d_keypoints_filename_label)
        controls_layout.addWidget(self.vid_3d_keypoints_filename_input)
        controls_layout.addWidget(self.vid_3d_save_video_cb)
        controls_layout.addWidget(self.vid_3d_output_video_filename_label)
        controls_layout.addWidget(self.vid_3d_output_video_filename_input)
        controls_layout.addWidget(self.vid_3d_save_video_black_background_cb)
        controls_layout.addWidget(self.vid_3d_output_video_black_background_filename_label)
        controls_layout.addWidget(self.vid_3d_output_video_black_background_filename_input)
        controls_layout.addWidget(self.vid_3d_send_keypoints_cb)
        controls_layout.addWidget(self.vid_3d_port_label)
        controls_layout.addWidget(self.vid_3d_port_input)
        # Add 3D webcam widgets to layout
        controls_layout.addWidget(self.cam_3d_plot_landmarks_cb)
        controls_layout.addWidget(self.cam_3d_plot_values_cb)
        controls_layout.addWidget(self.cam_3d_save_keypoints_cb)
        controls_layout.addWidget(self.cam_3d_keypoints_filename_label)
        controls_layout.addWidget(self.cam_3d_keypoints_filename_input)
        controls_layout.addWidget(self.cam_3d_save_video_cb)
        controls_layout.addWidget(self.cam_3d_output_video_filename_label)
        controls_layout.addWidget(self.cam_3d_output_video_filename_input)
        controls_layout.addWidget(self.cam_3d_save_video_black_background_cb)
        controls_layout.addWidget(self.cam_3d_output_video_black_background_filename_label)
        controls_layout.addWidget(self.cam_3d_output_video_black_background_filename_input)
        controls_layout.addWidget(self.cam_3d_send_keypoints_cb)
        controls_layout.addWidget(self.cam_3d_port_label)
        controls_layout.addWidget(self.cam_3d_port_input)
        # Add 3D phone widgets to layout
        controls_layout.addWidget(self.phone_3d_ip_label)
        controls_layout.addWidget(self.phone_3d_ip_input)
        controls_layout.addWidget(self.phone_3d_plot_landmarks_cb)
        controls_layout.addWidget(self.phone_3d_plot_values_cb)
        controls_layout.addWidget(self.phone_3d_save_keypoints_cb)
        controls_layout.addWidget(self.phone_3d_keypoints_filename_label)
        controls_layout.addWidget(self.phone_3d_keypoints_filename_input)
        controls_layout.addWidget(self.phone_3d_save_video_cb)
        controls_layout.addWidget(self.phone_3d_output_video_filename_label)
        controls_layout.addWidget(self.phone_3d_output_video_filename_input)
        controls_layout.addWidget(self.phone_3d_save_video_black_background_cb)
        controls_layout.addWidget(self.phone_3d_output_video_black_background_filename_label)
        controls_layout.addWidget(self.phone_3d_output_video_black_background_filename_input)
        controls_layout.addWidget(self.phone_3d_send_keypoints_cb)
        controls_layout.addWidget(self.phone_3d_port_label)
        controls_layout.addWidget(self.phone_3d_port_input)
        
        # Add 3D video with Depth Model widgets to layout
        
        controls_layout.addWidget(self.vid_3d_depth_browse_button)
        controls_layout.addWidget(self.vid_3d_depth_use_depth_model_cb)
        controls_layout.addWidget(self.vid_3d_depth_disp_depth_map_ch)
        controls_layout.addWidget(self.vid_3d_depth_plot_landmarks_cb)
        controls_layout.addWidget(self.vid_3d_depth_plot_values_cb)
        controls_layout.addWidget(self.vid_3d_depth_save_keypoints_cb)
        controls_layout.addWidget(self.vid_3d_depth_keypoints_filename_label)
        controls_layout.addWidget(self.vid_3d_depth_keypoints_filename_input)
        controls_layout.addWidget(self.vid_3d_depth_save_video_cb)
        controls_layout.addWidget(self.vid_3d_depth_output_video_filename_label)
        controls_layout.addWidget(self.vid_3d_depth_output_video_filename_input)
        controls_layout.addWidget(self.vid_3d_depth_save_video_black_background_cb)
        controls_layout.addWidget(self.vid_3d_depth_output_video_black_background_filename_label)
        controls_layout.addWidget(self.vid_3d_depth_output_video_black_background_filename_input)
        controls_layout.addWidget(self.vid_3d_depth_send_keypoints_cb)
        controls_layout.addWidget(self.vid_3d_depth_port_label)
        controls_layout.addWidget(self.vid_3d_depth_port_input)
        #endregion
        
        # Add a "spacer" to push all the controls to the top of the column
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        controls_layout.addItem(spacer)

        # --- Assemble the Main Layout ---
        # Create a vertical layout for the media display and status label
        media_layout = QVBoxLayout()
        media_layout.addWidget(self.displayed_media_label, 1) # Give it stretch factor

        # Add the media layout to the main layout (it will be the left column)
        main_layout.addLayout(media_layout, 3) # The '3' gives it more horizontal stretch space
        # Add the controls layout to the main layout (it will be the right column)
        main_layout.addLayout(controls_layout, 1) 

        # --- Set the main layout ---
        # --- Create a main layout and add the menu bar and the existing layout --- 
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setMenuBar(self.menu_bar)  # Add the menu bar to the layout
        self.main_layout.addLayout(main_layout)     # Add the rest of your UI below the menu bar
        self.setLayout(self.main_layout)

    def setup_worker_thread(self):
        """Creates and connects the worker thread."""
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        # Connect signals from this (main) thread to the worker's slots
        # 2D signals
        self.start_image_processing_signal.connect(self.worker.process_image)
        self.start_video_processing_signal.connect(self.worker.process_video)
        self.start_webcam_processing_signal.connect(self.worker.process_webcam)
        self.start_phone_processing_signal.connect(self.worker.process_phone_stream) 
        # 3D signals
        self.start_3d_video_processing_signal.connect(self.worker.process_3d_video)
        self.start_3d_webcam_processing_signal.connect(self.worker.process_3d_webcam)
        self.start_3d_phone_processing_signal.connect(self.worker.process_3d_phone)
        self.start_3d_depth_video_processing_signal.connect(self.worker.process_3d_video_with_depth_model)
        self.stop_worker_signal.connect(self.worker.stop) 
        self.switch_the_model_signal.connect(self.worker.switch_model)
        
        # Connect signals from the worker back to this (main) thread's slots
        self.worker.image_finished.connect(self.on_image_processing_finished)
        self.worker.video_finished.connect(self.on_video_processing_finished)
        self.worker.new_frame_ready.connect(self.display_processed_image) # Connect new frame signal
        self.worker.error.connect(self.on_processing_error)

        # Connect thread management signals
        self.thread.started.connect(lambda: print("Worker thread started."))
        self.thread.finished.connect(lambda: print("Worker thread finished."))
        self.thread.start()

    def on_dimension_changed(self, index):
        dimension = self.dropdown.currentText()
        # Hide everything to start fresh
        self.dropdown_2d.hide()
        self.dropdown_3d.hide()
        self.hide_all_2D_sources_options()
        self.hide_all_3D_sources_options()
        self.uploaded_image_path = None
        self.uploaded_video_path = None
        
        if dimension == '2D':
            self.dropdown_2d.show()
            self.dropdown_2d.setCurrentIndex(0)
        elif dimension == '3D':
            self.dropdown_3d.show()
            self.dropdown_3d.setCurrentIndex(0)

    def on_source_2D_changed(self, index):
        source = self.dropdown_2d.currentText()
        # Hide everything first to ensure a clean slate
        self.hide_all_2D_sources_options()

        if source == 'Upload Image':
            self.img_browse_button.show()
            self.displayed_media_label.setText('Select an image to process.')
        elif source == 'Upload Video':
            self.vid_browse_button.show()
            self.displayed_media_label.setText('Select a video to process.')
        elif source == 'Use Webcam':
            self.show_webcam_options()
            self.displayed_media_label.setText('Ready to start webcam. Press "Start".')
        elif source == 'Use Smartphone Camera':
            self.show_phone_options()
            self.displayed_media_label.setText('Enter your phone IP and press "Start".')
        else:
            self.displayed_media_label.setText('Your selected media will be displayed here.')

    def on_source_3D_changed(self, index):
        source = self.dropdown_3d.currentText()
        
        self.hide_all_3D_sources_options()
        
        if source == 'Upload Video':
            self.vid_3d_browse_button.show()
            self.displayed_media_label.setText('Select a video for 3D processing.')
        elif source == 'Use Webcam':
            self.show_3d_webcam_options()
            self.displayed_media_label.setText('Ready to start 3D Webcam. Press "Start".')
        elif source == 'Use Smartphone Camera':
            self.show_3d_phone_options() # To be implemented
            self.displayed_media_label.setText('3D Smartphone processing is not yet implemented.')
        elif source == 'Upload Video with Depth Model':
            self.vid_3d_depth_browse_button.show()
            self.displayed_media_label.setText('Select a video for 3D processing with Depth Model.')
        else:
            self.displayed_media_label.setText('Your selected media will be displayed here.')
    
    def open_image_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.uploaded_image_path = file_name
            self.uploaded_video_path = None # Clear video path
            
            # Load the image using our new orientation-aware function
            image_np = load_image_with_orientation(self.uploaded_image_path)

            if image_np is not None:
                # Use our updated display function to show the preview
                self.display_processed_image(image_np)
                self.show_image_options()
            else:
                QMessageBox.critical(self, "Error", f"Failed to load image file: {file_name}")
                self.displayed_media_label.setText("Could not load image.")
                # self.hide_all_source_options()

    def open_video_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.uploaded_video_path = file_name
            self.uploaded_image_path = None # Clear image path
            self.displayed_media_label.setText(f"Selected video:\n{file_name}")
            
            # This logic needs to adapt based on which mode is active
            if self.dropdown.currentText() == '2D':
                self.show_video_options()
            elif self.dropdown.currentText() == '3D':
                source = self.dropdown_3d.currentText()
                if source == 'Upload Video':
                    self.show_3d_video_options()
                elif source == 'Upload Video with Depth Model':
                    self.show_3d_depth_video_options()

    def start_processing(self):
        """Determines which processing to start based on dropdown selection."""
        dimension = self.dropdown.currentText()
        
        if dimension == '2D':
            self.start_2d_processing()
        elif dimension == '3D':
            self.start_3d_processing()

    def start_2d_processing(self):
        """Handles starting all 2D processing tasks."""
        source = self.dropdown_2d.currentText()
        if source == 'Upload Image':
            if self.uploaded_image_path:
                print("Starting image processing...")
                self.set_ui_enabled(False)
                
                # --- Get parameters from UI ---
                plot_landmarks = self.img_plot_landmarks_cb.isChecked()
                plot_skeleton = self.img_plot_skeleton_cb.isChecked()
                save_landmarks = self.img_save_landmarks_cb.isChecked()
                save_image = self.img_save_processed_image_cb.isChecked()
                save_image_black_background = self.img_save_processed_image_black_background_cb.isChecked()
                output_size_str = self.img_processed_image_size_combo.currentText()
                landmarks_filename = self.img_landmarks_filename_input.text()
                processed_image_filename = self.img_processed_image_filename_input.text()
                processed_image_black_background_filename = self.img_processed_image_black_background_filename_input.text()
                # --- Emit the signal to the worker thread ---
                self.start_image_processing_signal.emit(
                    self.uploaded_image_path,
                    plot_landmarks,
                    plot_skeleton,
                    save_landmarks,
                    landmarks_filename,
                    save_image,
                    output_size_str,
                    processed_image_filename,
                    save_image_black_background,
                    processed_image_black_background_filename
                )
            else:
                QMessageBox.warning(self, "Warning", "Please select an image first.")
        
        elif source == 'Upload Video':
            if self.uploaded_video_path:
                print("Starting video processing...")
                self.set_ui_enabled(False)

                # --- Get parameters from UI ---
                plot_landmarks = self.vid_plot_landmarks_cb.isChecked()
                plot_skeleton = self.vid_plot_skeleton_cb.isChecked()
                plot_values = self.vid_plot_values_cb.isChecked()
                save_landmarks = self.vid_save_landmarks_cb.isChecked()
                landmarks_filename = self.vid_landmarks_filename_input.text()
                save_video = self.vid_save_processed_video_cb.isChecked()
                output_video_filename = self.vid_output_video_filename_input.text()
                save_video_black_background = self.vid_save_processed_video_black_background_cb.isChecked()
                output_video_black_background_filename = self.vid_output_video_black_background_filename_input.text()
                # --- Emit the signal to the worker thread ---
                self.start_video_processing_signal.emit(
                    self.uploaded_video_path,
                    plot_landmarks,
                    plot_skeleton,
                    plot_values,
                    save_landmarks,
                    save_video,
                    landmarks_filename,
                    output_video_filename,
                    save_video_black_background,
                    output_video_black_background_filename
                )
            else:
                QMessageBox.warning(self, "Warning", "Please select a video first.")

        elif source == 'Use Webcam':
            print("Starting webcam processing...")
            self.set_ui_enabled(False)
            
            # --- Get parameters from UI ---
            plot_landmarks = self.cam_plot_landmarks_cb.isChecked()
            plot_skeleton = self.cam_plot_skeleton_cb.isChecked()
            plot_values = self.cam_plot_values_cb.isChecked()
            save_landmarks = self.cam_save_landmarks_cb.isChecked()
            landmarks_filename = self.cam_landmarks_filename_input.text()
            save_video = self.cam_save_processed_video_cb.isChecked()
            output_video_filename = self.cam_output_video_filename_input.text()
            save_video_black_background = self.cam_save_processed_video_black_background_cb.isChecked()
            output_video_black_background_filename = self.cam_output_video_black_background_filename_input.text()
            # --- Emit the signal to the worker thread ---
            self.start_webcam_processing_signal.emit(
                plot_landmarks,
                plot_skeleton,
                plot_values,
                save_landmarks,
                save_video,
                landmarks_filename,
                output_video_filename,
                save_video_black_background,
                output_video_black_background_filename
            )

        elif source == 'Use Smartphone Camera':
            phone_ip = self.phone_ip_input.text()
            if phone_ip:
                print("Starting phone camera processing...")
                self.set_ui_enabled(False)

                # --- Get parameters from UI ---
                plot_landmarks = self.phone_plot_landmarks_cb.isChecked()
                plot_skeleton = self.phone_plot_skeleton_cb.isChecked()
                plot_values = self.phone_plot_values_cb.isChecked()
                save_landmarks = self.phone_save_landmarks_cb.isChecked()
                landmarks_filename = self.phone_landmarks_filename_input.text()
                save_video = self.phone_save_processed_video_cb.isChecked()
                output_video_filename = self.phone_output_video_filename_input.text()
                save_video_black_background = self.phone_save_processed_video_black_background_cb.isChecked()
                output_video_black_background_filename = self.phone_output_video_black_background_filename_input.text()
                # --- Emit the signal to the worker thread ---
                self.start_phone_processing_signal.emit(
                    phone_ip,
                    plot_landmarks,
                    plot_skeleton,
                    plot_values,
                    save_landmarks,
                    save_video,
                    landmarks_filename,
                    output_video_filename,
                    save_video_black_background,
                    output_video_black_background_filename
                )
            else:
                QMessageBox.warning(self, "Warning", "Please enter your phone's IP address.")

    def start_3d_processing(self):
        """Handles starting all 3D processing tasks."""
        source = self.dropdown_3d.currentText()
        if source == 'Upload Video':
            if self.uploaded_video_path:

                send_keypoints = self.vid_3d_send_keypoints_cb.isChecked()
                port = 0
                if send_keypoints:
                    try:
                        port = int(self.vid_3d_port_input.text())
                    except ValueError:
                        QMessageBox.warning(self, "Warning", "Please enter a valid integer for the port.")
                        return

                print("Starting 3D video processing...")
                self.set_ui_enabled(False)

                # Get parameters from UI
                plot_landmarks_skeleton = self.vid_3d_plot_landmarks_cb.isChecked()
                plot_values = self.vid_3d_plot_values_cb.isChecked()
                save_keypoints = self.vid_3d_save_keypoints_cb.isChecked()
                keypoints_filename = self.vid_3d_keypoints_filename_input.text()
                save_video = self.vid_3d_save_video_cb.isChecked()
                video_filename = self.vid_3d_output_video_filename_input.text()
                save_video_black = self.vid_3d_save_video_black_background_cb.isChecked()
                video_filename_black = self.vid_3d_output_video_black_background_filename_input.text()
                
                self.start_3d_video_processing_signal.emit(
                    self.uploaded_video_path,
                    plot_landmarks_skeleton,
                    plot_values,
                    save_keypoints,
                    keypoints_filename,
                    save_video,
                    video_filename,
                    save_video_black,
                    video_filename_black,
                    send_keypoints,
                    port
                )
            else:
                QMessageBox.warning(self, "Warning", "Please select a video first.")
        elif source == 'Use Webcam':

            send_keypoints = self.cam_3d_send_keypoints_cb.isChecked()
            port = 0
            if send_keypoints:
                try:
                    port = int(self.cam_3d_port_input.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Please enter a valid integer for the port.")
                    return
            
            print("Starting 3D fixed webcam processing...")
            self.set_ui_enabled(False)

            plot_landmarks_skeleton = self.cam_3d_plot_landmarks_cb.isChecked()
            plot_values = self.cam_3d_plot_values_cb.isChecked()
            save_keypoints = self.cam_3d_save_keypoints_cb.isChecked()
            keypoints_filename = self.cam_3d_keypoints_filename_input.text()
            save_video = self.cam_3d_save_video_cb.isChecked()
            video_filename = self.cam_3d_output_video_filename_input.text()
            save_video_black = self.cam_3d_save_video_black_background_cb.isChecked()
            video_filename_black = self.cam_3d_output_video_black_background_filename_input.text()

            self.start_3d_webcam_processing_signal.emit(
                plot_landmarks_skeleton,
                plot_values,
                save_keypoints,
                keypoints_filename,
                save_video,
                video_filename,
                save_video_black,
                video_filename_black,
                send_keypoints,
                port
            )
        elif source == 'Use Smartphone Camera':
            phone_ip = self.phone_3d_ip_input.text()
            if not phone_ip:
                QMessageBox.warning(self, "Warning", "Please enter the phone's IP address.")
                return

            send_keypoints = self.phone_3d_send_keypoints_cb.isChecked()
            port = 0
            if send_keypoints:
                try:
                    port = int(self.phone_3d_port_input.text())
                except ValueError:
                    QMessageBox.warning(self, "Warning", "Please enter a valid integer for the port.")
                    return
            
            print("Starting 3D fixed phone processing...")
            self.set_ui_enabled(False)
            
            plot_landmarks_skeleton = self.phone_3d_plot_landmarks_cb.isChecked()
            plot_values = self.phone_3d_plot_values_cb.isChecked()
            save_keypoints = self.phone_3d_save_keypoints_cb.isChecked()
            keypoints_filename = self.phone_3d_keypoints_filename_input.text()
            save_video = self.phone_3d_save_video_cb.isChecked()
            video_filename = self.phone_3d_output_video_filename_input.text()
            save_video_black = self.phone_3d_save_video_black_background_cb.isChecked()
            video_filename_black = self.phone_3d_output_video_black_background_filename_input.text()

            self.start_3d_phone_processing_signal.emit(
                phone_ip,
                plot_landmarks_skeleton,
                plot_values,
                save_keypoints,
                keypoints_filename,
                save_video,
                video_filename,
                save_video_black,
                video_filename_black,
                send_keypoints,
                port
            )
        elif source == 'Upload Video with Depth Model':
            if self.uploaded_video_path:

                send_keypoints = self.vid_3d_depth_send_keypoints_cb.isChecked()
                port = 0
                if send_keypoints:
                    try:
                        port = int(self.vid_3d_depth_port_input.text())
                    except ValueError:
                        QMessageBox.warning(self, "Warning", "Please enter a valid integer for the port.")
                        return

                print("Starting 3D video processing with Depth Model...")
                self.set_ui_enabled(False)

                # Get parameters from UI
                disp_depth_map = self.vid_3d_depth_disp_depth_map_ch.isChecked()
                use_depth_model = self.vid_3d_depth_use_depth_model_cb.isChecked()
                plot_landmarks_skeleton = self.vid_3d_depth_plot_landmarks_cb.isChecked()
                plot_values = self.vid_3d_depth_plot_values_cb.isChecked()
                save_keypoints = self.vid_3d_depth_save_keypoints_cb.isChecked()
                keypoints_filename = self.vid_3d_depth_keypoints_filename_input.text()
                save_video = self.vid_3d_depth_save_video_cb.isChecked()
                video_filename = self.vid_3d_depth_output_video_filename_input.text()
                save_video_black = self.vid_3d_depth_save_video_black_background_cb.isChecked()
                video_filename_black = self.vid_3d_depth_output_video_black_background_filename_input.text()
                
                self.start_3d_depth_video_processing_signal.emit(
                    self.uploaded_video_path,
                    use_depth_model,
                    disp_depth_map,
                    plot_landmarks_skeleton,
                    plot_values,
                    save_keypoints,
                    keypoints_filename,
                    save_video,
                    video_filename,
                    save_video_black,
                    video_filename_black,
                    send_keypoints,
                    port
                )
            else:
                QMessageBox.warning(self, "Warning", "Please select a video first.")

    def stop_processing(self):
        """Emits the signal to stop the worker's process."""
        print("Stop button clicked. Emitting stop signal to worker.")
        self.stop_worker_signal.emit()
        self.dropdown_2d.setCurrentIndex(0)
        self.dropdown_3d.setCurrentIndex(0)
        self.status_label.setText("")
        self.set_ui_enabled(True) # Re-enable UI immediately

    def display_processed_image(self, image_np):
        """Displays a numpy image in the media label, scaling it efficiently."""
        try:
            # Convert from BGR (OpenCV) to RGB (Qt)
            image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            height, width, channel = image_np_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_np_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # --- Smart Scaling ---
            # Scale the pixmap to fit the label's dimensions while preserving aspect ratio.
            # This is more efficient than setScaledContents(True) for large images.
            scaled_pixmap = pixmap.scaled(self.displayed_media_label.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.displayed_media_label.setPixmap(scaled_pixmap)
            self.displayed_media_label.setScaledContents(False) 
            self.displayed_media_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            print(f"Error displaying image: {e}")
            self.displayed_media_label.setText("Error displaying frame.")

    def set_style_for_widgets(self, widget, font_size, color, background_color=None, tooltip=None, cursor_shape=Qt.CursorShape.PointingHandCursor, border_radius=15, padding=8):
        widget.setFont(QFont("Arial", font_size))
        widget.setCursor(cursor_shape)
        if background_color:
            widget.setStyleSheet(f"background-color:{background_color}; color: {color}; border-radius: {border_radius}px; padding: {padding}px;")
        else:
            widget.setStyleSheet(f"color: {color}; border-radius: {border_radius}px; padding: {padding}px;")
        if tooltip:
            widget.setToolTip(tooltip)
    
    def on_image_processing_finished(self, processed_image_np):
        """Handles the completion of image processing."""
        print("Image processing finished.")
        self.display_processed_image(processed_image_np)
        self.set_ui_enabled(True)

    def on_video_processing_finished(self, message):
        """Handles the completion of video processing."""
        print(f"Video processing finished: {message}")
        self.set_ui_enabled(True)
        QMessageBox.information(self, "Info", message)

    def on_processing_error(self, error_message):
        """Handles errors reported by the worker."""
        print(f"An error occurred: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)
        self.set_ui_enabled(True)

    def on_image_save_landmarks_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.img_landmarks_filename_label.setVisible(is_checked)
        self.img_landmarks_filename_input.setVisible(is_checked)
        
    def on_image_save_processed_image_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.img_processed_image_filename_label.setVisible(is_checked)
        self.img_processed_image_filename_input.setVisible(is_checked)
        self.img_processed_image_size_label.setVisible(is_checked)
        self.img_processed_image_size_combo.setVisible(is_checked)

    def on_image_save_processed_image_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.img_processed_image_black_background_filename_label.setVisible(is_checked)
        self.img_processed_image_black_background_filename_input.setVisible(is_checked)
        self.img_processed_image_size_label.setVisible(is_checked)
        self.img_processed_image_size_combo.setVisible(is_checked)

    def on_video_save_landmarks_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_landmarks_filename_label.setVisible(is_checked)
        self.vid_landmarks_filename_input.setVisible(is_checked)

    def on_video_save_processed_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_output_video_filename_label.setVisible(is_checked)
        self.vid_output_video_filename_input.setVisible(is_checked)

    def on_video_save_processed_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_output_video_black_background_filename_label.setVisible(is_checked)
        self.vid_output_video_black_background_filename_input.setVisible(is_checked)

    def on_webcam_save_landmarks_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_landmarks_filename_label.setVisible(is_checked)
        self.cam_landmarks_filename_input.setVisible(is_checked)

    def on_webcam_save_processed_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_output_video_filename_label.setVisible(is_checked)
        self.cam_output_video_filename_input.setVisible(is_checked)

    def on_webcam_save_processed_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_output_video_black_background_filename_label.setVisible(is_checked)
        self.cam_output_video_black_background_filename_input.setVisible(is_checked)
    
    def on_phone_save_landmarks_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_landmarks_filename_label.setVisible(is_checked)
        self.phone_landmarks_filename_input.setVisible(is_checked)

    def on_phone_save_processed_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_output_video_filename_label.setVisible(is_checked)
        self.phone_output_video_filename_input.setVisible(is_checked)

    def on_phone_save_processed_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_output_video_black_background_filename_label.setVisible(is_checked)
        self.phone_output_video_black_background_filename_input.setVisible(is_checked)

    def show_image_options(self):
        self.img_plot_landmarks_cb.show()
        self.img_plot_skeleton_cb.show()
        self.img_save_landmarks_cb.show()
        self.on_image_save_landmarks_changed(self.img_save_landmarks_cb.checkState().value)
        self.img_save_processed_image_cb.show()
        self.img_save_processed_image_black_background_cb.show()
        self.on_image_save_processed_image_changed(self.img_save_processed_image_cb.checkState().value)
        self.on_image_save_processed_image_black_background_changed(self.img_save_processed_image_black_background_cb.checkState().value)
        
    def show_video_options(self):
        self.vid_plot_landmarks_cb.show()
        self.vid_plot_skeleton_cb.show()
        self.vid_plot_values_cb.show()
        self.vid_save_landmarks_cb.show()
        self.on_video_save_landmarks_changed(self.vid_save_landmarks_cb.checkState().value)
        self.vid_save_processed_video_cb.show()
        self.on_video_save_processed_video_changed(self.vid_save_processed_video_cb.checkState().value)
        self.vid_save_processed_video_black_background_cb.show()
        self.on_video_save_processed_video_black_background_changed(self.vid_save_processed_video_black_background_cb.checkState().value)

    def show_webcam_options(self):
        self.cam_plot_landmarks_cb.show()
        self.cam_plot_skeleton_cb.show()
        self.cam_plot_values_cb.show()
        self.cam_save_landmarks_cb.show()
        self.on_webcam_save_landmarks_changed(self.cam_save_landmarks_cb.checkState().value)
        self.cam_save_processed_video_cb.show()
        self.on_webcam_save_processed_video_changed(self.cam_save_processed_video_cb.checkState().value)
        self.cam_save_processed_video_black_background_cb.show()
        self.on_webcam_save_processed_video_black_background_changed(self.cam_save_processed_video_black_background_cb.checkState().value)

    def show_phone_options(self):
        self.phone_ip_label.show()
        self.phone_ip_input.show()
        self.phone_plot_landmarks_cb.show()
        self.phone_plot_skeleton_cb.show()
        self.phone_plot_values_cb.show()
        self.phone_save_landmarks_cb.show()
        self.on_phone_save_landmarks_changed(self.phone_save_landmarks_cb.checkState().value)
        self.phone_save_processed_video_cb.show()
        self.on_phone_save_processed_video_changed(self.phone_save_processed_video_cb.checkState().value)
        self.phone_save_processed_video_black_background_cb.show()
        self.on_phone_save_processed_video_black_background_changed(self.phone_save_processed_video_black_background_cb.checkState().value)
        
    def show_3d_video_options(self):
        self.vid_3d_browse_button.show()
        self.vid_3d_plot_landmarks_cb.show()
        self.vid_3d_plot_values_cb.show()
        self.vid_3d_save_keypoints_cb.show()
        self.on_3d_video_save_keypoints_changed(self.vid_3d_save_keypoints_cb.checkState().value)
        self.vid_3d_save_video_cb.show()
        self.on_3d_video_save_video_changed(self.vid_3d_save_video_cb.checkState().value)
        self.vid_3d_save_video_black_background_cb.show()
        self.on_3d_video_save_video_black_background_changed(self.vid_3d_save_video_black_background_cb.checkState().value)
        self.vid_3d_send_keypoints_cb.show()
        self.on_3d_video_send_keypoints_changed(self.vid_3d_send_keypoints_cb.checkState().value)

    def show_3d_depth_video_options(self):
        self.vid_3d_depth_browse_button.show()
        self.vid_3d_depth_disp_depth_map_ch.show()
        self.vid_3d_depth_use_depth_model_cb.show()
        self.vid_3d_depth_plot_landmarks_cb.show()
        self.vid_3d_depth_plot_values_cb.show()
        self.vid_3d_depth_save_keypoints_cb.show()
        self.on_3d_depth_video_save_keypoints_changed(self.vid_3d_depth_save_keypoints_cb.checkState().value)
        self.vid_3d_depth_save_video_cb.show()
        self.on_3d_depth_video_save_video_changed(self.vid_3d_depth_save_video_cb.checkState().value)
        self.vid_3d_depth_save_video_black_background_cb.show()
        self.on_3d_depth_video_save_video_black_background_changed(self.vid_3d_save_video_black_background_cb.checkState().value)
        self.vid_3d_depth_send_keypoints_cb.show()
        self.on_3d_depth_video_send_keypoints_changed(self.vid_3d_send_keypoints_cb.checkState().value)

    def show_3d_webcam_options(self):
        self.cam_3d_plot_landmarks_cb.show()
        self.cam_3d_plot_values_cb.show()
        self.cam_3d_save_keypoints_cb.show()
        self.on_3d_cam_save_keypoints_changed(self.cam_3d_save_keypoints_cb.checkState().value)
        self.cam_3d_save_video_cb.show()
        self.on_3d_cam_save_video_changed(self.cam_3d_save_video_cb.checkState().value)
        self.cam_3d_save_video_black_background_cb.show()
        self.on_3d_cam_save_video_black_background_changed(self.cam_3d_save_video_black_background_cb.checkState().value)
        self.cam_3d_send_keypoints_cb.show()
        self.on_3d_cam_send_keypoints_changed(self.cam_3d_send_keypoints_cb.checkState().value)

    def hide_all_2D_sources_options(self):
        self.hide_image_options()
        self.hide_video_options()
        self.hide_webcam_options()
        self.hide_phone_options()
    
    def hide_all_3D_sources_options(self):
        self.hide_3d_video_options()
        self.hide_3d_webcam_options()
        self.hide_3d_phone_options()
        self.hide_3d_depth_video_options()

    def hide_image_options(self):
        self.img_browse_button.hide()
        self.img_plot_landmarks_cb.hide()
        self.img_plot_skeleton_cb.hide()
        self.img_save_landmarks_cb.hide()
        self.img_landmarks_filename_label.hide()
        self.img_landmarks_filename_input.hide()
        self.img_save_processed_image_cb.hide()
        self.img_processed_image_filename_label.hide()
        self.img_processed_image_filename_input.hide()
        self.img_processed_image_size_label.hide()
        self.img_processed_image_size_combo.hide()
        self.img_save_processed_image_black_background_cb.hide()
        self.img_processed_image_black_background_filename_label.hide()
        self.img_processed_image_black_background_filename_input.hide()

    def hide_video_options(self):
        self.vid_browse_button.hide()
        self.vid_plot_landmarks_cb.hide()
        self.vid_plot_skeleton_cb.hide()
        self.vid_plot_values_cb.hide()
        self.vid_save_landmarks_cb.hide()
        self.vid_landmarks_filename_label.hide()
        self.vid_landmarks_filename_input.hide()
        self.vid_save_processed_video_cb.hide()
        self.vid_output_video_filename_label.hide()
        self.vid_output_video_filename_input.hide()
        self.vid_save_processed_video_black_background_cb.hide()
        self.vid_output_video_black_background_filename_label.hide()
        self.vid_output_video_black_background_filename_input.hide()

    def hide_webcam_options(self):
        self.cam_plot_landmarks_cb.hide()
        self.cam_plot_skeleton_cb.hide()
        self.cam_plot_values_cb.hide()
        self.cam_save_landmarks_cb.hide()
        self.cam_landmarks_filename_label.hide()
        self.cam_landmarks_filename_input.hide()
        self.cam_save_processed_video_cb.hide()
        self.cam_output_video_filename_label.hide()
        self.cam_output_video_filename_input.hide()
        self.cam_save_processed_video_black_background_cb.hide()
        self.cam_output_video_black_background_filename_label.hide()
        self.cam_output_video_black_background_filename_input.hide()

    def hide_phone_options(self):
        self.phone_ip_label.hide()
        self.phone_ip_input.hide()
        self.phone_plot_landmarks_cb.hide()
        self.phone_plot_skeleton_cb.hide()
        self.phone_plot_values_cb.hide()
        self.phone_save_landmarks_cb.hide()
        self.phone_landmarks_filename_label.hide()
        self.phone_landmarks_filename_input.hide()
        self.phone_save_processed_video_cb.hide()
        self.phone_output_video_filename_label.hide()
        self.phone_output_video_filename_input.hide()
        self.phone_save_processed_video_black_background_cb.hide()
        self.phone_output_video_black_background_filename_label.hide()
        self.phone_output_video_black_background_filename_input.hide()

    def hide_3d_video_options(self):
        self.vid_3d_browse_button.hide()
        self.vid_3d_plot_landmarks_cb.hide()
        self.vid_3d_plot_values_cb.hide()
        self.vid_3d_save_keypoints_cb.hide()
        self.vid_3d_keypoints_filename_label.hide()
        self.vid_3d_keypoints_filename_input.hide()
        self.vid_3d_save_video_cb.hide()
        self.vid_3d_output_video_filename_label.hide()
        self.vid_3d_output_video_filename_input.hide()
        self.vid_3d_save_video_black_background_cb.hide()
        self.vid_3d_output_video_black_background_filename_label.hide()
        self.vid_3d_output_video_black_background_filename_input.hide()
        self.vid_3d_send_keypoints_cb.hide()
        self.vid_3d_port_label.hide()
        self.vid_3d_port_input.hide()
        
    def hide_3d_depth_video_options(self):
        self.vid_3d_depth_browse_button.hide()
        self.vid_3d_depth_disp_depth_map_ch.hide()
        self.vid_3d_depth_use_depth_model_cb.hide()
        self.vid_3d_depth_plot_landmarks_cb.hide()
        self.vid_3d_depth_plot_values_cb.hide()
        self.vid_3d_depth_save_keypoints_cb.hide()
        self.vid_3d_depth_keypoints_filename_label.hide()
        self.vid_3d_depth_keypoints_filename_input.hide()
        self.vid_3d_depth_save_video_cb.hide()
        self.vid_3d_depth_output_video_filename_label.hide()
        self.vid_3d_depth_output_video_filename_input.hide()
        self.vid_3d_depth_save_video_black_background_cb.hide()
        self.vid_3d_depth_output_video_black_background_filename_label.hide()
        self.vid_3d_depth_output_video_black_background_filename_input.hide()
        self.vid_3d_depth_send_keypoints_cb.hide()
        self.vid_3d_depth_port_label.hide()
        self.vid_3d_depth_port_input.hide()

    def hide_3d_webcam_options(self):
        self.cam_3d_plot_landmarks_cb.hide()
        self.cam_3d_plot_values_cb.hide()
        self.cam_3d_save_keypoints_cb.hide()
        self.cam_3d_keypoints_filename_label.hide()
        self.cam_3d_keypoints_filename_input.hide()
        self.cam_3d_save_video_cb.hide()
        self.cam_3d_output_video_filename_label.hide()
        self.cam_3d_output_video_filename_input.hide()
        self.cam_3d_save_video_black_background_cb.hide()
        self.cam_3d_output_video_black_background_filename_label.hide()
        self.cam_3d_output_video_black_background_filename_input.hide()
        self.cam_3d_send_keypoints_cb.hide()
        self.cam_3d_port_label.hide()
        self.cam_3d_port_input.hide()

    def hide_3d_phone_options(self):
        self.phone_3d_ip_label.hide()
        self.phone_3d_ip_input.hide()
        self.phone_3d_plot_landmarks_cb.hide()
        self.phone_3d_plot_values_cb.hide()
        self.phone_3d_save_keypoints_cb.hide()
        self.phone_3d_keypoints_filename_label.hide()
        self.phone_3d_keypoints_filename_input.hide()
        self.phone_3d_save_video_cb.hide()
        self.phone_3d_output_video_filename_label.hide()
        self.phone_3d_output_video_filename_input.hide()
        self.phone_3d_save_video_black_background_cb.hide()
        self.phone_3d_output_video_black_background_filename_label.hide()
        self.phone_3d_output_video_black_background_filename_input.hide()
        self.phone_3d_send_keypoints_cb.hide()
        self.phone_3d_port_label.hide()
        self.phone_3d_port_input.hide()

    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements to prevent user interaction during processing."""
        # --- Disable/Enable buttons ---
        self.start_button.setEnabled(enabled)
        # --- Disable/Enable dropdowns ---
        self.dropdown.setEnabled(enabled)
        self.dropdown_2d.setEnabled(enabled)
        self.dropdown_3d.setEnabled(enabled)
        # --- Disable/Enable all other controls ---
        # image
        self.img_browse_button.setEnabled(enabled)
        self.img_plot_landmarks_cb.setEnabled(enabled)
        self.img_plot_skeleton_cb.setEnabled(enabled)
        self.img_save_landmarks_cb.setEnabled(enabled)
        self.img_landmarks_filename_input.setEnabled(enabled)
        self.img_save_processed_image_cb.setEnabled(enabled)
        self.img_processed_image_size_combo.setEnabled(enabled)
        self.img_processed_image_filename_input.setEnabled(enabled)
        self.img_save_processed_image_black_background_cb.setEnabled(enabled)

        # video
        self.vid_browse_button.setEnabled(enabled)
        self.vid_plot_landmarks_cb.setEnabled(enabled)
        self.vid_plot_skeleton_cb.setEnabled(enabled)
        self.vid_plot_values_cb.setEnabled(enabled)
        self.vid_save_landmarks_cb.setEnabled(enabled)
        self.vid_landmarks_filename_input.setEnabled(enabled)
        self.vid_save_processed_video_cb.setEnabled(enabled)
        self.vid_output_video_filename_input.setEnabled(enabled)
        self.vid_save_processed_video_black_background_cb.setEnabled(enabled)

        # webcam
        self.cam_plot_landmarks_cb.setEnabled(enabled)
        self.cam_plot_skeleton_cb.setEnabled(enabled)
        self.cam_plot_values_cb.setEnabled(enabled)
        self.cam_save_landmarks_cb.setEnabled(enabled)
        self.cam_landmarks_filename_input.setEnabled(enabled)
        self.cam_save_processed_video_cb.setEnabled(enabled)
        self.cam_output_video_filename_input.setEnabled(enabled)
        self.cam_save_processed_video_black_background_cb.setEnabled(enabled)

        # phone
        self.phone_ip_input.setEnabled(enabled)
        self.phone_plot_landmarks_cb.setEnabled(enabled)
        self.phone_plot_skeleton_cb.setEnabled(enabled)
        self.phone_plot_values_cb.setEnabled(enabled)
        self.phone_save_landmarks_cb.setEnabled(enabled)
        self.phone_landmarks_filename_input.setEnabled(enabled)
        self.phone_save_processed_video_cb.setEnabled(enabled)
        self.phone_output_video_filename_input.setEnabled(enabled)
        self.phone_save_processed_video_black_background_cb.setEnabled(enabled)

        # 3d video
        self.vid_3d_plot_landmarks_cb.setEnabled(enabled)
        self.vid_3d_plot_values_cb.setEnabled(enabled)
        self.vid_3d_save_keypoints_cb.setEnabled(enabled)
        self.vid_3d_keypoints_filename_input.setEnabled(enabled)
        self.vid_3d_save_video_cb.setEnabled(enabled)
        self.vid_3d_output_video_filename_input.setEnabled(enabled)
        self.vid_3d_save_video_black_background_cb.setEnabled(enabled)
        self.vid_3d_output_video_black_background_filename_input.setEnabled(enabled)
        self.vid_3d_send_keypoints_cb.setEnabled(enabled)
        self.vid_3d_port_input.setEnabled(enabled)
        # 3d webcam
        self.cam_3d_plot_landmarks_cb.setEnabled(enabled)
        self.cam_3d_plot_values_cb.setEnabled(enabled)
        self.cam_3d_save_keypoints_cb.setEnabled(enabled)
        self.cam_3d_keypoints_filename_input.setEnabled(enabled)
        self.cam_3d_save_video_cb.setEnabled(enabled)
        self.cam_3d_output_video_filename_input.setEnabled(enabled)
        self.cam_3d_save_video_black_background_cb.setEnabled(enabled)
        self.cam_3d_output_video_black_background_filename_input.setEnabled(enabled)
        self.cam_3d_send_keypoints_cb.setEnabled(enabled)
        self.cam_3d_port_input.setEnabled(enabled)
        
        # 3d phone
        self.phone_3d_ip_input.setEnabled(enabled)
        self.phone_3d_plot_landmarks_cb.setEnabled(enabled)
        self.phone_3d_plot_values_cb.setEnabled(enabled)
        self.phone_3d_save_keypoints_cb.setEnabled(enabled)
        self.phone_3d_keypoints_filename_input.setEnabled(enabled)
        self.phone_3d_save_video_cb.setEnabled(enabled)
        self.phone_3d_output_video_filename_input.setEnabled(enabled)
        self.phone_3d_save_video_black_background_cb.setEnabled(enabled)
        self.phone_3d_output_video_black_background_filename_input.setEnabled(enabled)
        self.phone_3d_send_keypoints_cb.setEnabled(enabled)
        self.phone_3d_port_input.setEnabled(enabled)
        
        # 3d video with Depth Model
        self.vid_3d_depth_browse_button.setEnabled(enabled)
        self.vid_3d_depth_disp_depth_map_ch.setEnabled(enabled)
        self.vid_3d_depth_use_depth_model_cb.setEnabled(enabled)
        self.vid_3d_depth_plot_landmarks_cb.setEnabled(enabled)
        self.vid_3d_depth_plot_values_cb.setEnabled(enabled)
        self.vid_3d_depth_save_keypoints_cb.setEnabled(enabled)
        self.vid_3d_depth_keypoints_filename_input.setEnabled(enabled)
        self.vid_3d_depth_save_video_cb.setEnabled(enabled)
        self.vid_3d_depth_output_video_filename_input.setEnabled(enabled)
        self.vid_3d_depth_save_video_black_background_cb.setEnabled(enabled)
        self.vid_3d_depth_output_video_black_background_filename_input.setEnabled(enabled)
        self.vid_3d_depth_send_keypoints_cb.setEnabled(enabled)
        self.vid_3d_depth_port_input.setEnabled(enabled)
        
        # --- Change button appearance based on state ---
        if enabled:
            self.start_button.setStyleSheet("background-color:rgb(117, 248, 121); color: white; border-radius: 15px; padding: 8px;")
            self.end_button.setStyleSheet("background-color: #f44336; color: white; border-radius: 15px; padding: 8px;")
        else:
            # A "disabled" look
            self.start_button.setStyleSheet("background-color: #cccccc; color: #666666; border: 1px solid #999999; border-radius: 15px; padding: 8px;")
            self.end_button.setStyleSheet("background-color: #f44336; color: white; border-radius: 15px; padding: 8px;") # End remains active
            
        # The End button is special. It should be ENABLED when processing is running
        self.end_button.setEnabled(not enabled)

    def closeEvent(self, event):
        """Ensure the worker thread is properly shut down when the window closes."""
        print("Main window is closing. Stopping worker thread...")
        self.stop_worker_signal.emit() # Tell the worker to stop any processing
        self.thread.quit() # Ask the thread's event loop to exit
        if not self.thread.wait(5000): # Wait up to 5 seconds for the thread to finish
            print("Warning: Worker thread did not terminate gracefully. Forcing termination.")
            self.thread.terminate()
            self.thread.wait() # Wait again after forced termination
        
        print("Thread stopped.")
        event.accept()

    def on_3d_video_save_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_keypoints_filename_label.setVisible(is_checked)
        self.vid_3d_keypoints_filename_input.setVisible(is_checked)
    
    def on_3d_depth_video_save_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_depth_keypoints_filename_label.setVisible(is_checked)
        self.vid_3d_depth_keypoints_filename_input.setVisible(is_checked)
    
    def on_3d_video_save_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_output_video_filename_label.setVisible(is_checked)
        self.vid_3d_output_video_filename_input.setVisible(is_checked)
        
    def on_3d_depth_video_save_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_depth_output_video_filename_label.setVisible(is_checked)
        self.vid_3d_depth_output_video_filename_input.setVisible(is_checked)

    def on_3d_video_save_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_output_video_black_background_filename_label.setVisible(is_checked)
        self.vid_3d_output_video_black_background_filename_input.setVisible(is_checked)
    
    def on_3d_depth_video_save_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_depth_output_video_black_background_filename_label.setVisible(is_checked)
        self.vid_3d_depth_output_video_black_background_filename_input.setVisible(is_checked)

    def on_3d_video_send_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_port_label.setVisible(is_checked)
        self.vid_3d_port_input.setVisible(is_checked)
    
    def on_3d_depth_video_send_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.vid_3d_depth_port_label.setVisible(is_checked)
        self.vid_3d_depth_port_input.setVisible(is_checked)

    def on_3d_cam_save_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_3d_keypoints_filename_label.setVisible(is_checked)
        self.cam_3d_keypoints_filename_input.setVisible(is_checked)
        
    def on_3d_cam_save_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_3d_output_video_filename_label.setVisible(is_checked)
        self.cam_3d_output_video_filename_input.setVisible(is_checked)

    def on_3d_cam_save_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_3d_output_video_black_background_filename_label.setVisible(is_checked)
        self.cam_3d_output_video_black_background_filename_input.setVisible(is_checked)

    def on_3d_cam_send_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.cam_3d_port_label.setVisible(is_checked)
        self.cam_3d_port_input.setVisible(is_checked)

    def on_3d_phone_save_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_3d_keypoints_filename_label.setVisible(is_checked)
        self.phone_3d_keypoints_filename_input.setVisible(is_checked)
        
    def on_3d_phone_save_video_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_3d_output_video_filename_label.setVisible(is_checked)
        self.phone_3d_output_video_filename_input.setVisible(is_checked)

    def on_3d_phone_save_video_black_background_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_3d_output_video_black_background_filename_label.setVisible(is_checked)
        self.phone_3d_output_video_black_background_filename_input.setVisible(is_checked)

    def on_3d_phone_send_keypoints_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.phone_3d_port_label.setVisible(is_checked)
        self.phone_3d_port_input.setVisible(is_checked)

    def show_3d_phone_options(self):
        self.phone_3d_ip_label.show()
        self.phone_3d_ip_input.show()
        # self.phone_3d_scaling_time_label.show()
        # self.phone_3d_scaling_time_input.show()
        self.phone_3d_plot_landmarks_cb.show()
        self.phone_3d_plot_values_cb.show()
        self.phone_3d_save_keypoints_cb.show()
        self.on_3d_phone_save_keypoints_changed(self.phone_3d_save_keypoints_cb.checkState().value)
        self.phone_3d_save_video_cb.show()
        self.on_3d_phone_save_video_changed(self.phone_3d_save_video_cb.checkState().value)
        self.phone_3d_save_video_black_background_cb.show()
        self.on_3d_phone_save_video_black_background_changed(self.phone_3d_save_video_black_background_cb.checkState().value)
        self.phone_3d_send_keypoints_cb.show()
        self.on_3d_phone_send_keypoints_changed(self.phone_3d_send_keypoints_cb.checkState().value)

    # --- For Styling the APP ---
    def apply_theme(self, theme='light'):
        """Apply the selected theme to the entire application"""
        if theme == 'light':
            self.setStyleSheet(UIStyles.LIGHT_THEME)
        else:
            self.setStyleSheet(UIStyles.DARK_THEME)
        
        # Set object names for special styling
        self.start_button.setObjectName('start_button')
        self.end_button.setObjectName('end_button')
        self.displayed_media_label.setObjectName('displayed_media_label')
        self.status_label.setObjectName('status_label')

    def set_light_mode(self):
        """Switch to light theme"""
        self.apply_theme('light')

    def set_dark_mode(self):
        """Switch to dark theme"""
        self.apply_theme('dark')

    # --- For Switching Models ---
    def set_light_model(self):
        """Switch to light model"""
        self.switch_the_model_signal.emit(1)
    
    def set_heavy_model(self):
        """Switch to heavy model"""
        self.switch_the_model_signal.emit(2)
    
    # --- Dialog Methods ---
    def show_about_dialog(self):
        about_text = """
        <h2>About the 3D Human Motion Capture System</h2>
        <p>This application allows you to perform advanced 2D and 3D human motion capture from various sources. Here’s a breakdown of its features:</p>
        
        <h3>Processing Modes:</h3>
        <ul>
            <li><b>2D Processing:</b> Extracts and visualizes 2D landmarks (keypoints) of a person's body from an image, video, webcam feed, or a phone camera stream. It can draw the skeleton, save the keypoints, and save the processed media.</li>
            <li><b>3D Processing:</b> Captures 3D landmarks to understand body motion in three-dimensional space. This is perfect for applications in animation, biomechanics, and virtual reality.</li>
        </ul>

        <h3>Media Sources:</h3>
        <ul>
            <li><b>Image:</b> Upload a static image (like a JPG or PNG) to analyze a single pose.</li>
            <li><b>Video:</b> Upload a pre-recorded video file to process a sequence of movements.</li>
            <li><b>Webcam:</b> Use your computer's built-in webcam for real-time motion capture.</li>
            <li><b>Phone Camera:</b> Stream live video from your phone's camera to the application for real-time processing.</li>
        </ul>

        <h3>Key Features:</h3>
        <ul>
            <li><b>Landmark Detection:</b> Uses Google's MediaPipe library to accurately detect 33 body landmarks.</li>
            <li><b>Skeleton Visualization:</b> Draws lines connecting the landmarks to visualize the human skeleton.</li>
            <li><b>Data Export:</b> You can save the detected keypoints as a JSON file, which is useful for analysis or for use in other applications (like game engines).</li>
            <li><b>Real-time Streaming:</b> In 3D mode, the system can broadcast the keypoint data over a WebSocket, allowing other applications (like Unity) to receive the motion data in real-time.</li>
        </ul>
        """
        QMessageBox.about(self, "About This Application", about_text)

    def show_contact_dialog(self):
        contact_text = """
        <b>Aleppo, Syria</b><br>
        <b>Phone:</b> +963 991292874<br>
        <b>Email:</b> alharth.alhaj.hussein@gmail.com<br>
        <b>LinkedIn:</b> <a href='https://www.linkedin.com/in/alharth-alhaj-hussein-023417241/'>https://www.linkedin.com/in/alharth-alhaj-hussein-023417241/</a><br>
        <b>GitHub:</b> <a href='https://github.com/AlharthAlhajHussein?tab=repositories'>https://github.com/AlharthAlhajHussein?tab=repositories</a><br>
        <b>Portfolio:</b> <a href='https://alharths-data-canvas.lovable.app/'>https://alharths-data-canvas.lovable.app/</a>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Contact Us")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        msg_box.setText(contact_text)
        msg_box.exec()

    










