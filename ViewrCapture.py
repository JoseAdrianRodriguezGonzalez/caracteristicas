# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:10:23 2024

Captura nubes filtradas - se agrego la ventana de visualizacion de profundidad y color. 
Captura una nube de puntos al presionar q y la mostrara filtrada y sin filtrar, las 
lineas 121 y 122 establecen la distancia de captura
Linea 49 es la carpeta donde se guardan los datos
Presiona ESC para cerrar la ventana

@author:
"""
import os
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from datetime import datetime

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Output directory for point cloud data
OUTPUT_PATH = 'captured_data'

# Delete dir if exists
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
    
# Create timestamp dir
now = datetime.now()
path = f"{OUTPUT_PATH}/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
os.mkdir(path)

# Streaming loop
frame_count = 0

try:
    while True:
        # Wait for frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # If resolutions are different, resize color image
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # Press 'q' to capture point cloud and save images
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            # Save color and depth images
            cv2.imwrite(f'{path}/prueba_profundidad.png', depth_colormap)
            cv2.imwrite(f'{path}/prueba_color.png', color_image)

            # Create point cloud
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            vtx = np.asanyarray(points.get_vertices())
            xyz = np.asanyarray(vtx).view(np.float32).reshape(-1, 3)

            # Adjust coordinates
            xyz[:, 1] *= -1
            xyz[:, 2] *= -1

            # Save raw point cloud data
            np.savetxt(f"{path}/capture_{frame_count}.csv", xyz, delimiter=",")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Display the point cloud object
            o3d.visualization.draw_geometries([pcd])
            
            # Filter based on distance
            xyz_norm = np.linalg.norm(xyz, axis=1)
            min_dist = 0.00
            max_dist = 1.00
            mask = (xyz_norm > min_dist) & (xyz_norm < max_dist)
            xyz_filtered = xyz[mask]

            # Save filtered data
            np.savetxt(f"{path}/capture_filtered_{frame_count}.csv", xyz_filtered, delimiter=",")

            # Create Open3D point cloud object for filtered data
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_filtered)

            # Display the filtered point cloud
            o3d.visualization.draw_geometries([pcd])

            frame_count += 1

        # Press 'Esc' to exit
        if key == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close OpenCV window
    cv2.destroyAllWindows()
