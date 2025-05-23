# Exercise Form Checker

An AI-powered computer vision system that evaluates workout form, specifically focusing on squat exercises. The system uses machine learning to analyze knee angles and provide real-time feedback on exercise form.

## Features

- Real-time knee angle analysis during squats
- Machine learning-based form classification
- Support for both left and right knee tracking
- Visual feedback with decision boundary plots
- Cross-validation for model accuracy
- Standardized data preprocessing pipeline

## Prerequisites

- Python 3.8 or higher
- Webcam or video input device
- Sufficient lighting for accurate pose detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Exercise-Form-Checker.git
cd Exercise-Form-Checker
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Exercise-Form-Checker/
├── Assets/                  # Media and resource files
├── data_visualization/      # Visualization scripts and outputs
├── src/                     # Source code
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks for analysis
├── min_knee_angle_classifier.py    # Main classifier implementation
├── min_knee_angle_metrics.py       # Metrics calculation
├── enhanced_train.py              # Training script
└── requirements.txt               # Project dependencies
```

## Usage

1. Run the knee angle classifier:
```bash
python min_knee_angle_classifier.py
```

2. For training the model with custom data:
```bash
python enhanced_train.py
```

## Model Details

The system uses a Support Vector Machine (SVM) classifier with RBF kernel to analyze minimum knee angles during squats. The model is trained on features including:
- Minimum left knee angle
- Minimum right knee angle

The model provides binary classification (valid/invalid) for squat form based on these metrics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- MediaPipe for pose detection
- OpenCV for computer vision capabilities
- scikit-learn for machine learning implementation
