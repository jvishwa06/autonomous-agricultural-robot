# Autonomous Agriculture Robot

This project focuses on the development of an autonomous agriculture robot designed to efficiently detect, pick, and transport paddy to a storage zone. Leveraging AI, ROS, and state-of-the-art hardware, the robot is optimized for real-world agricultural applications.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Hardware Used](#hardware-used)
- [Software Requirements](#software-requirements)
- [Model Training and Optimization](#model-training-and-optimization)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This autonomous agriculture robot was developed to automate the process of paddy harvesting. It uses advanced AI and ROS capabilities for precise paddy detection, picking, and transportation. The robot is equipped with a trained and quantized object detection model deployed on a Jetson Orin Nano and utilizes an Intel RealSense depth camera for point cloud generation and navigation.

## Features

- **AI-driven Paddy Detection:** Trained object detection model optimized for real-time paddy identification.
- **Efficient Harvesting:** The robot autonomously detects, picks, and transports paddy to the storage zone.
- **Camera Calibration:** Accurate camera calibration for accurate depth values.
- **Depth Sensing:** Utilization of Intel RealSense depth camera for obtaining point clouds and enhanced object detection.
- **Optimized Performance:** Deployment of the quantized model on Jetson Orin Nano for high efficiency.

## Hardware Used

- **Jetson Orin Nano:** Used for deploying the AI model and controlling the robot.
- **Intel RealSense Depth Camera:** Employed for capturing depth data and generating point clouds.
- **Autonomous Navigation System:** For guiding the robot through the field.

## Software Requirements

- **Ubuntu 20.04 LTS**
- **ROS Noetic**
- **Python 3.8+**
- **PyTorch**
- **Intel RealSense SDK**
- **CUDA (for Jetson Orin Nano)**
- **OpenCV**
- **YOLOv8**



## Model Training and Optimization

- **Training:** The object detection model was trained using YOLOv8 on a dataset of paddy images.
- **Quantization:** Post-training, the model was quantized to FP16 for optimized deployment on Jetson Orin Nano.
- **Calibration:** Camera calibration was performed to ensure accurate depth measurement and navigation.

## Performance Metrics

- **Accuracy:** The trained model achieved an accuracy of 95% on the validation dataset.
- **Latency:** The inference time on Jetson Orin Nano was reduced to  15 ms after quantization.
- **Navigation Precision:** The robot successfully navigated and harvested paddy with a precision of 80%.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. For major changes, please open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to the ROS community for providing extensive documentation and support.
- Acknowledgment to the creators of YOLOv8 for their contribution to the field of computer vision.
