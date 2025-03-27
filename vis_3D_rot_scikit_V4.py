#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMU Visualization Tool for Biomechanical Analysis
-----------------------------------------------
This script visualizes the orientation of multiple IMUs (Inertial Measurement Units)
using quaternion data. IMUs are commonly used in biomechanics to measure segment
orientations during movement.

Key Biomechanical Concepts:
1. IMU Orientation: Represented using quaternions [w, x, y, z]
   - w: Scalar component (cos(θ/2))
   - x, y, z: Vector components (axis of rotation * sin(θ/2))
   - Quaternions avoid gimbal lock and provide smooth interpolation

2. Coordinate System:
   - X-axis (Red): Points forward along the long axis of the IMU
   - Y-axis (Green): Points to the left
   - Z-axis (Blue): Points upward
   This follows the standard biomechanical convention for segment orientation.

3. Sampling Rate:
   - Original: 120 Hz (typical for biomechanical analysis)
   - Downsampled: 30 Hz for visualization
   - This provides sufficient temporal resolution for human movement analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from pathlib import Path

class IMUVisualizer:
    def __init__(self, csv_files, plot_freq=30):
        """
        Initialize the IMU visualizer.
        
        Parameters:
        -----------
        csv_files : list
            List of paths to IMU data files (Xsens DOT format)
        plot_freq : int
            Display frequency in Hz (default: 30 Hz)
            
        Biomechanical Note:
        -------------------
        30 Hz display rate is sufficient for human movement visualization,
        as human eye can typically process 24-30 frames per second.
        """
        self.plot_freq = plot_freq
        self.csv_files = csv_files
        self.fig = None
        self.axes = None
        self.lines = None
        self.quats = None
        self.times = None
        self.frame_texts = None
        
    def read_data(self):
        """
        Read and process IMU data from CSV files.
        
        Biomechanical Notes:
        -------------------
        1. Xsens DOT CSV Format:
           - First 7 rows: Metadata (device info, firmware, etc.)
           - Header row: Column names
           - Data rows: Time series of quaternions and sensor data
        
        2. Quaternion Order:
           - [w, x, y, z] format
           - w: Scalar component (cos(θ/2))
           - x, y, z: Vector components (axis of rotation * sin(θ/2))
        
        3. Time Processing:
           - SampleTimeFine: Microsecond precision
           - Converted to seconds for easier interpretation
           - Relative time (zeroed at start) for movement analysis
        """
        try:
            self.quats = []
            self.times = []
            
            for csv_file in self.csv_files:
                # Read the CSV file, skipping metadata but keeping header
                df = pd.read_csv(csv_file, skiprows=7)
                
                # Extract quaternions in [w, x, y, z] order
                quats = df[['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']].values
                
                # Get timestamps (convert to seconds)
                times = df['SampleTimeFine'].values
                times = (times - times[0]) / 1e6  # Convert to seconds
                
                self.quats.append(quats)
                self.times.append(times)
            
            # Calculate downsampling factor based on the first IMU
            original_freq = 1 / (self.times[0][1] - self.times[0][0])
            downsample_factor = max(1, int(original_freq / self.plot_freq))
            
            # Find minimum length after downsampling
            min_length = min(len(quats[::downsample_factor]) for quats in self.quats)
            
            # Downsample data for all IMUs and ensure same length
            for i in range(len(self.quats)):
                self.quats[i] = self.quats[i][::downsample_factor][:min_length]
                self.times[i] = self.times[i][::downsample_factor][:min_length]
            
            print(f"Original sampling rate: {original_freq:.1f} Hz")
            print(f"Downsampled to: {self.plot_freq} Hz")
            print(f"Data duration: {self.times[0][-1] - self.times[0][0]:.2f} seconds")
            print(f"Number of samples: {len(self.times[0])}")
            print(f"Initial quaternions:")
            for i, csv_file in enumerate(self.csv_files):
                print(f"{Path(csv_file).stem}: {self.quats[i][0]}")
            
        except Exception as e:
            print(f"Error reading data: {e}")
            raise
            
    def create_cuboid(self):
        """
        Create a 3D cuboid representation of the IMU.
        
        Biomechanical Notes:
        -------------------
        1. Cuboid Dimensions:
           - Longer in X direction (1.2 units) to show forward orientation
           - Shorter in Y and Z (0.6 and 0.4 units) for realistic proportions
           - These dimensions are arbitrary but maintain aspect ratio
        
        2. Coordinate Axes:
           - X-axis (Red): Forward direction (primary axis)
           - Y-axis (Green): Left direction
           - Z-axis (Blue): Upward direction
           This follows standard biomechanical conventions for segment orientation.
        """
        # Create a rectangular cuboid (longer in X direction to show forward)
        vertices = np.array([
            [-0.6, -0.3, -0.2],  # 0 - back, bottom, left
            [0.6, -0.3, -0.2],   # 1 - front, bottom, left
            [0.6, 0.3, -0.2],    # 2 - front, bottom, right
            [-0.6, 0.3, -0.2],   # 3 - back, bottom, right
            [-0.6, -0.3, 0.2],   # 4 - back, top, left
            [0.6, -0.3, 0.2],    # 5 - front, top, left
            [0.6, 0.3, 0.2],     # 6 - front, top, right
            [-0.6, 0.3, 0.2]     # 7 - back, top, right
        ])
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Add direction indicators
        # X-axis arrow (red)
        vertices = np.vstack((vertices, [
            [0, 0, 0],      # Origin
            [0.8, 0, 0],    # X-axis
        ]))
        edges.append((8, 9))  # X-axis edge
        
        # Y-axis arrow (green)
        vertices = np.vstack((vertices, [
            [0, 0.5, 0],    # Y-axis
        ]))
        edges.append((8, 10))  # Y-axis edge
        
        # Z-axis arrow (blue)
        vertices = np.vstack((vertices, [
            [0, 0, 0.4],    # Z-axis
        ]))
        edges.append((8, 11))  # Z-axis edge
        
        return vertices, edges
        
    def setup_plot(self):
        """
        Initialize the 3D visualization layout.
        
        Biomechanical Notes:
        -------------------
        1. Subplot Layout:
           - 2x2 grid for 4 IMUs
           - Each subplot shows independent coordinate system
           - Allows comparison of segment orientations
        
        2. View Settings:
           - elev=20, azim=45: Standard biomechanical viewing angle
           - Equal aspect ratio ensures accurate orientation representation
           - Axis limits [-1, 1] provide consistent scale across all IMUs
        """
        self.fig = plt.figure(figsize=(15, 15))
        
        # Create 2x2 subplot grid
        self.axes = []
        self.lines = []
        self.frame_texts = []
        
        # Define color schemes for each IMU
        color_schemes = {
            'LF': {
                'cuboid': '#808080',  # Gray
                'x_axis': '#ff0000',  # Red
                'y_axis': '#00ff00',  # Green
                'z_axis': '#0000ff'   # Blue
            },
            'LH': {
                'cuboid': '#808080',  # Gray
                'x_axis': '#ff0000',  # Red
                'y_axis': '#00ff00',  # Green
                'z_axis': '#0000ff'   # Blue
            },
            'LSh': {
                'cuboid': '#808080',  # Gray
                'x_axis': '#ff0000',  # Red
                'y_axis': '#00ff00',  # Green
                'z_axis': '#0000ff'   # Blue
            },
            'LT': {
                'cuboid': '#808080',  # Gray
                'x_axis': '#ff0000',  # Red
                'y_axis': '#00ff00',  # Green
                'z_axis': '#0000ff'   # Blue
            }
        }
        
        # Create subplots for each IMU
        for i, csv_file in enumerate(self.csv_files):
            # Create subplot
            ax = self.fig.add_subplot(2, 2, i+1, projection='3d')
            self.axes.append(ax)
            
            # Get color scheme for this IMU
            imu_name = Path(csv_file).stem.split('_')[0]  # Get IMU name without timestamp
            colors = color_schemes[imu_name]
            
            # Create cuboid
            vertices, edges = self.create_cuboid()
            
            # Create lines for this IMU
            imu_lines = []
            for j, (start, end) in enumerate(edges):
                if j < 8:  # Main cuboid edges
                    line, = ax.plot([], [], [], color=colors['cuboid'], linestyle='-', linewidth=2)
                elif j == 8:  # X-axis
                    line, = ax.plot([], [], [], color=colors['x_axis'], linestyle='-', linewidth=2)
                elif j == 9:  # Y-axis
                    line, = ax.plot([], [], [], color=colors['y_axis'], linestyle='-', linewidth=2)
                else:  # Z-axis
                    line, = ax.plot([], [], [], color=colors['z_axis'], linestyle='-', linewidth=2)
                imu_lines.append(line)
            self.lines.append(imu_lines)
            
            # Add IMU label
            ax.text(0, 0.5, 0.4, imu_name, color=colors['cuboid'], fontsize=12, fontweight='bold',
                   horizontalalignment='center', verticalalignment='center')
            
            # Add time text
            frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
            self.frame_texts.append(frame_text)
            
            # Set axis properties
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Left)')
            ax.set_zlabel('Z (Up)')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Set axis limits
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            
            # Set initial view
            ax.view_init(elev=20, azim=45)
        
        # Adjust layout
        plt.tight_layout()
        
    def update(self, frame):
        """
        Update function for animation.
        
        Biomechanical Notes:
        -------------------
        1. Quaternion to Rotation:
           - scipy.spatial.transform.Rotation handles quaternion conversion
           - Applies rotation to all vertices of the cuboid
           - Maintains proper orientation relationships
        
        2. Time Display:
           - Shows elapsed time in seconds
           - Helps correlate movement phases
           - Useful for gait analysis and movement timing
        """
        try:
            # Update each IMU
            for i in range(len(self.csv_files)):
                # Get current quaternion [w, x, y, z]
                qr, qx, qy, qz = self.quats[i][frame]
                
                # Create rotation object for current orientation
                r = Rotation.from_quat([qx, qy, qz, qr])
                
                # Get vertices and edges
                vertices, edges = self.create_cuboid()
                
                # Rotate vertices
                rotated_vertices = r.apply(vertices)
                
                # Update lines for this IMU
                for j, (start, end) in enumerate(edges):
                    self.lines[i][j].set_data([rotated_vertices[start, 0], rotated_vertices[end, 0]],
                                            [rotated_vertices[start, 1], rotated_vertices[end, 1]])
                    self.lines[i][j].set_3d_properties([rotated_vertices[start, 2], rotated_vertices[end, 2]])
                
                # Update time text
                self.frame_texts[i].set_text(f'Time: {self.times[i][frame]:.2f}s')
            
            return [line for imu_lines in self.lines for line in imu_lines] + self.frame_texts
            
        except Exception as e:
            print(f"Error in update: {e}")
            plt.close()
            raise
            
    def run(self):
        """
        Run the visualization.
        
        Biomechanical Notes:
        -------------------
        1. Animation Settings:
           - 30 Hz display rate matches typical video frame rate
           - interval=1000/self.plot_freq ensures smooth animation
           - blit=True improves performance for real-time visualization
        
        2. Data Processing:
           - Downsampling reduces computational load
           - Maintains sufficient temporal resolution for movement analysis
           - Synchronizes multiple IMUs for coordinated visualization
        """
        try:
            # Read the data
            self.read_data()
            
            # Setup the plot
            self.setup_plot()
            
            # Create animation
            anim = FuncAnimation(
                self.fig,
                self.update,
                frames=len(self.times[0]),
                interval=1000/self.plot_freq,  # Convert to milliseconds
                blit=True,
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            print(f"Error running visualization: {e}")
            raise

def main():
    """
    Main function to run the visualization.
    
    Biomechanical Notes:
    -------------------
    IMU Placement:
    - LF: Left Foot
    - LH: Left Hand
    - LSh: Left Shoulder
    - LT: Left Thigh
    
    These placements allow analysis of:
    - Gait patterns (foot movement)
    - Upper limb kinematics (hand and shoulder)
    - Lower limb kinematics (thigh movement)
    """
    # Create visualizer with 30 Hz display rate
    csv_files = [
        'data/rec/LF_20250211_121711.csv',
        'data/rec/LH_20250211_121711.csv',
        'data/rec/LSh_20250211_121711.csv',
        'data/rec/LT_20250211_121711.csv'
    ]
    viz = IMUVisualizer(csv_files, plot_freq=30)
    
    # Run visualization
    viz.run()

if __name__ == '__main__':
    main() 