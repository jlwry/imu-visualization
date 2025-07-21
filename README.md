# IMU Visualization Tool
<img width="1247" height="704" alt="image" src="https://github.com/user-attachments/assets/0693c39a-e13b-4f9d-af46-5cf8b74ea492" />



A Python-based tool for visualizing IMU (Inertial Measurement Unit) orientation data in 3D space. This tool is specifically designed for biomechanical analysis, allowing visualization of multiple IMUs simultaneously.

## Features

- Real-time 3D visualization of IMU orientations
- Support for multiple IMUs (up to 4) in separate subplots
- Standard biomechanical coordinate system representation
- Comprehensive documentation of biomechanical concepts
- Smooth animation at 30 Hz display rate

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/imu-visualization.git
cd imu-visualization
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib scipy
```

## Usage

1. Place your IMU data files in the `data/rec/` directory
2. Run the visualization script:
```bash
python vis_3D_rot_scikit_V4.py
```

## Data Format

The tool expects CSV files in Xsens DOT format with the following structure:
- First 7 rows: Metadata
- Header row: Column names
- Data rows: Time series of quaternions and sensor data

## Coordinate System

- X-axis (Red): Forward direction
- Y-axis (Green): Left direction
- Z-axis (Blue): Upward direction
