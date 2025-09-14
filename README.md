# 🤸‍♂️ Human Motion Capture System

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-Aleppo-yellow.svg)](https://opensource.org/licenses/Aleppo)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

An advanced desktop application for real-time 2D and 3D human pose estimation, built with PyQt6, OpenCV, [MediaPipe](https://github.com/google-ai-edge/mediapipe), and [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) and more.

<br>

## 🎥 Project Demo Video

Watch this short video to see the system's key features in action!

[![Watch the demo](https://github.com/AlharthAlhajHussein/Human-Motion-Capture-System/blob/main/images/video_icon.png)](https://youtu.be/nQ9ILd5VCtU)

Click the image to watch a demo video of the application in action.

<a href="https://youtu.be/nQ9ILd5VCtU" target="_blank">
  <img src="https://i.imgur.com/your_thumbnail_image.jpg" alt="Project Demo Video" width="600" />
</a>

<br>

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#️-usage)
- [Contributors](#-contributors)
- [Contributing](#-Contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 About The Project

This system is a powerful tool designed for analyzing human motion from various media sources. Whether you are a game developer, a biomechanics researcher, or an animator, this software provides the tools necessary to accurately capture and export motion data.

The system combines the speed and precision of Google's MediaPipe library for landmark detection with the computational power of the Depth Anything V2 model for 3D depth estimation. This results in realistic and reliable 3D motion tracking.

---

## ✨ Key Features

-   **2D and 3D Processing**: Choose between extracting keypoints in 2D or 3D space.
-   **Multiple Source Support**:
    -   Analyze static **Images**.
    -   Process pre-recorded **Video** files.
    -   Capture motion in real-time from a **Webcam**.
    -   Stream live video from a **Smartphone Camera**.
-   **Advanced Depth Estimation**: Utilizes the Depth Anything V2 model to enhance the accuracy of the Z-axis coordinate, providing realistic tracking of movement towards and away from the camera with displaying the depth map.
-   **Real-time Data Streaming**: Ability to broadcast 3D keypoint data over a WebSocket in real-time to other applications like Unity or Unreal Engine or any env that contains websocket protocol to receive keypoints.
-   **Data Export**:
    -   Save keypoints in a `JSON` file for later analysis.
    -   Save the processed video or image with or without background also with or without the skeleton overlay.
-   **Flexible User Interface**: An easy-to-use graphical interface with multiple customization options, MediaPipe (light or heavy) model, Depth Anything (small, base, large) model, including white and dark themes.

---

## 🛠️ Tech Stack

-   **User Interface**: PyQt6 
-   **Image & Video Processing**: OpenCV 
-   **Human Pose Estimation**: Google MediaPipe
-   **Depth Estimation (3D)**: PyTorch & Depth Anything V2
-   **Numerical Operations**: NumPy 
-   **Network Streaming**: WebSockets

---

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

-   Python 3.9 or newer
-   Git for version control

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/AlharthAlhajHussein/Human-Motion-Capture-System.git
    cd Human-Motion-Capture-System
    ```

2.  **Create and activate a virtual environment (recommended)**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```

4.  **⚠️ Important: Download the Depth Model**

    This project uses the **Depth Anything V2** model for accurate 3D depth estimation. You must download one of the pre-trained models and place it in the correct folder.

    -   **First, create the necessary directories:**
        Create a folder named `checkpoints` inside the `logic` folder. The final path should be: `logic/checkpoints/`.

    -   **Second, choose and download one of the following models:**

        -   **Small Model (vits)**: Fastest and least resource-intensive, suitable for quick tests.
            -   [Download Link for Small Model (vits)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)

        -   **Base Model (vitb)**: A good balance between speed and accuracy.
            -   [Download Link for Base Model (vitb)](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true)

        -   **Large Model (vitl)**: Most accurate but slowest, ideal for non-real-time processing.
            -   [Download Link for Large Model (vitl)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)

    -   **Finally, place the downloaded `.pth` file inside the `logic/checkpoints/` directory**. Ensure the filename matches the model selected in the code (e.g., `depth_anything_v2_vits.pth`).
    -   **Note:** if you want to run the depth model on your Nvidia GPU:
        - install [CUDA](https://developer.nvidia.com/cuda-downloads) that available for your GPU.
        - install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support. 

---

## ▶️ Usage

After completing all installation steps, you can run the application with the following command from the project's root directory:

```bash
python main.py
```
The main window will appear, and from there you can select the processing type and source to begin capturing motion.

---
## 👥 Contributors

I would like to thank everyone who contributed to the development of this project:

- [Zakaria Dliwati](https://github.com/AlharthAlhajHussein) 

- [Muhammad Ali](https://github.com/AlharthAlhajHussein)

---
## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/AlharthAlhajHussein/Human-Motion-Capture-System/issues) if you want to contribute.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---
📄 License

This project is licensed under the ALEPPO License. See the LICENSE file for more details.
---

## 👨‍💻 Author

**Alharth Alhaj Hussein**

📬 Connect with me:
-   [![Email](https://img.shields.io/badge/Gmail-0A66C2?style=for-the-badge&logo=gmail&logoColor=white)](https://www.linkedin.com/in/alharth-alhaj-hussein-023417241): **alharth.alhaj.hussein@gmail.com**
-   [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alharth-alhaj-hussein-023417241)
-   [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlharthAlhajHussein)
-   [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/alharthalhajhussein)
-   [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Alharth.Alhaj.Hussein)
---

Feel free to reach out with any questions or suggestions.

If you find this project insightful or useful, please consider giving it a ⭐ on GitHub!
