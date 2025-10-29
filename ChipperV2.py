import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import os
import shutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QLineEdit, QPushButton, QComboBox, 
                             QSpinBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import rasterio
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
import cv2

def get_starts(img_size, chip_size):
    if img_size <= 0:
        return []
    starts = list(range(0, img_size - chip_size + 1, chip_size))
    if starts and starts[-1] + chip_size < img_size:
        starts.append(starts[-1] + chip_size)
    elif not starts:
        starts = [0]
    return starts

def extract_image_chips(image_path, output_folder, chip_height=256, chip_width=256, output_format='png'):
    if output_format not in ['png', 'jpg']:
        raise ValueError("output_format must be 'png' or 'jpg'.")
    
    with rasterio.open(image_path) as src:
        img_data = src.read()
        img_height, img_width = src.height, src.width
        
        # Handle any number of bands: take up to 3, replicate last if fewer
        num_bands = len(img_data)
        if num_bands == 0:
            raise ValueError("No bands in TIFF.")
        if num_bands >= 3:
            img = img_data[:3]
        else:
            # Replicate last band to make 3 channels
            extra = np.repeat(img_data[-1:], 3 - num_bands, axis=0)
            img = np.concatenate([img_data, extra], axis=0)
        
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        
        # Convert to 8-bit: linear stretch to [0,255] uint8 per band
        for band in range(img.shape[2]):
            band_data = img[:, :, band]
            if band_data.max() > band_data.min():
                band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min()) * 255.0
            else:
                band_data = np.zeros_like(band_data)
            img[:, :, band] = np.clip(band_data, 0, 255).astype(np.uint8)
        
        img = img.astype(np.uint8)
    
    os.makedirs(output_folder, exist_ok=True)
    
    step_h = chip_height
    step_w = chip_width
    
    height_starts = get_starts(img_height, chip_height)
    width_starts = get_starts(img_width, chip_width)
    
    if not height_starts or not width_starts:
        raise ValueError("No chips can be extracted: image too small for chip size.")
    
    chip_id = 0
    for start_h in height_starts:
        for start_w in width_starts:
            end_h = min(start_h + chip_height, img_height)
            end_w = min(start_w + chip_width, img_width)
            chip = img[start_h:end_h, start_w:end_w]
            
            pad_h = chip_height - chip.shape[0]
            pad_w = chip_width - chip.shape[1]
            if pad_h > 0 or pad_w > 0:
                chip = np.pad(chip, ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='constant', constant_values=0)
            
            ext = output_format if output_format == 'png' else 'jpeg'
            chip_filename = f"chip_{chip_id:04d}.{output_format}"
            chip_path = os.path.join(output_folder, chip_filename)
            imsave(chip_path, chip, format=ext)
            chip_id += 1
    
    print(f"Extracted {chip_id} chips to {output_folder}")
    if chip_id == 0:
        raise ValueError("No chips were extracted.")
    return output_folder, height_starts, width_starts

def run_yolo_inference(model_path, source_folder, project_path, experiment_name, imgsz=1024, conf=0.1, return_results=False):
    if not os.path.exists(source_folder) or not os.listdir(source_folder):
        raise ValueError(f"Source folder empty or missing: {source_folder}")
    model = YOLO(model_path)
    results = model.predict(
        source=source_folder, 
        imgsz=imgsz, 
        conf=conf, 
        project=project_path, 
        name=experiment_name, 
        batch=4,
        save=True, 
        show_labels=False
    )
    if return_results:
        return list(results)
    print(f"YOLO inference completed. Results saved to {os.path.join(project_path, experiment_name)}")

def post_process(image_path, results_list, project_path, experiment_name, chip_height, height_starts, width_starts):
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        img_height, img_width = src.height, src.width
        
        if crs is None:
            print("Warning: Input TIFF has no CRS. Shapefile will use pixel coordinates.")
        
        # Reload full_img as 8-bit for visualization
        img_data = src.read()
        num_bands = len(img_data)
        if num_bands >= 3:
            img = img_data[:3]
        else:
            extra = np.repeat(img_data[-1:], 3 - num_bands, axis=0)
            img = np.concatenate([img_data, extra], axis=0)
        full_img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        for band in range(full_img.shape[2]):
            band_data = full_img[:, :, band]
            if band_data.max() > band_data.min():
                band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min()) * 255.0
            full_img[:, :, band] = np.clip(band_data, 0, 255).astype(np.uint8)
        full_img = full_img.astype(np.uint8)
        
        detections = []
    
    num_cols = len(width_starts)
    for r in results_list:
        if r.obb is None:
            continue
        chip_filename = os.path.basename(r.path)
        if not chip_filename.startswith('chip_') or not chip_filename.endswith('.jpg'):
            continue
        chip_id_str = chip_filename[5:-4]
        try:
            chip_id = int(chip_id_str)
        except ValueError:
            continue
        row = chip_id // num_cols
        col = chip_id % num_cols
        if row >= len(height_starts) or col >= len(width_starts):
            continue
        start_h = height_starts[row]
        start_w = width_starts[col]
        
        viz_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
        
        for i in range(len(r.obb)):
            box = r.obb.xyxyxyxy[i].cpu().numpy()
            global_box = (box + np.array([start_w, start_h])).flatten()
            pts = box.reshape(-1, 1, 2).astype(np.int32) + np.array([start_w, start_h])
            cv2.polylines(viz_img, [pts], True, (0, 255, 0), 2)
            
            points = []
            for j in range(4):
                col_ = box[j, 0] + start_w
                row_ = box[j, 1] + start_h
                geo_x, geo_y = rasterio.transform.xy(transform, row_, col_)
                points.append((geo_x, geo_y))
            poly = Polygon(points)
            detections.append({
                'conf': r.obb.conf[i].item(),
                'class': r.names[int(r.obb.cls[i].item())],
                'geometry': poly
            })
    
    annotated_path = os.path.join(project_path, f"{experiment_name}_annotated.jpg")
    cv2.imwrite(annotated_path, viz_img)
    print(f"Annotated image saved to {annotated_path}")
    
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    shp_path = os.path.join(project_path, f"{image_basename}.shp")
    if detections:
        gdf = gpd.GeoDataFrame(detections, crs=crs)
        gdf.to_file(shp_path)
        print(f"Shapefile saved to {shp_path} ({len(detections)} detections) with CRS {crs}")
    else:
        print("No detections found.")
    
    return os.path.join(project_path, "chips", experiment_name), os.path.join(project_path, experiment_name), annotated_path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ship Detection Inferencer")
        self.setGeometry(100, 100, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Image Path
        layout.addWidget(QLabel("Image Path (TIFF):"))
        self.image_path_edit = QLineEdit()
        browse_hbox = QHBoxLayout()
        browse_hbox.addWidget(self.image_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_image)
        browse_hbox.addWidget(browse_btn)
        layout.addLayout(browse_hbox)
        
        # Project Path
        layout.addWidget(QLabel("Project Path:"))
        self.project_path_edit = QLineEdit()
        project_hbox = QHBoxLayout()
        project_hbox.addWidget(self.project_path_edit)
        project_btn = QPushButton("Browse")
        project_btn.clicked.connect(self.browse_project)
        project_hbox.addWidget(project_btn)
        layout.addLayout(project_hbox)
        
        # Experiment Name
        layout.addWidget(QLabel("Experiment Name:"))
        self.experiment_name_edit = QLineEdit("experiment_1")
        layout.addWidget(self.experiment_name_edit)
        
        # Model Type
        layout.addWidget(QLabel("Model Type:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Sentinel-1", "Sentinel-2"])
        layout.addWidget(self.model_combo)
        
        # Chip Size
        layout.addWidget(QLabel("Chip Size (square):"))
        self.chip_size_spin = QSpinBox()
        self.chip_size_spin.setRange(64, 4096)
        self.chip_size_spin.setValue(1024)
        layout.addWidget(self.chip_size_spin)
        
        # Run Button
        self.run_btn = QPushButton("Run the Model")
        self.run_btn.clicked.connect(self.run_model)
        layout.addWidget(self.run_btn)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TIFF Image", "", "TIFF Files (*.tif *.tiff)")
        if file_path:
            self.image_path_edit.setText(file_path)
    
    def browse_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_path_edit.setText(dir_path)
    
    def run_model(self):
        image_path = self.image_path_edit.text().strip()
        project_path = self.project_path_edit.text().strip()
        experiment_name = self.experiment_name_edit.text().strip()
        model_type = self.model_combo.currentText()
        chip_size = self.chip_size_spin.value()
        
        if not all([image_path, project_path, experiment_name]):
            QMessageBox.warning(self, "Input Error", "Please fill all required fields.")
            return
        
        if not os.path.exists(image_path) or not image_path.lower().endswith(('.tif', '.tiff')):
            QMessageBox.warning(self, "Input Error", "Invalid image path: Must be a TIFF file.")
            return
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(script_dir, "models")
        
        if model_type == "Sentinel-1":
            model_path = os.path.join(models_folder, "sentinel1", "best.pt")
        else:
            model_path = os.path.join(models_folder, "sentinel2", "best.pt")
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", f"Model not found: {model_path}")
            return
        
        try:
            self.status_label.setText("Extracting chips...")
            self.run_btn.setEnabled(False)
            QApplication.processEvents()
            
            chips_dir = os.path.join(project_path, "chips")
            os.makedirs(chips_dir, exist_ok=True)
            chips_subdir = os.path.join(chips_dir, experiment_name)
            
            chips_folder, height_starts, width_starts = extract_image_chips(
                image_path=image_path,
                output_folder=chips_subdir,
                chip_height=chip_size,
                chip_width=chip_size,
                output_format='jpg'
            )
            
            self.status_label.setText("Running YOLO inference...")
            QApplication.processEvents()
            
            results_list = run_yolo_inference(
                model_path=model_path,
                source_folder=chips_subdir,
                project_path=project_path,
                experiment_name=experiment_name,
                imgsz=chip_size,
                conf=0.1,
                return_results=True
            )
            
            self.status_label.setText("Stitching and exporting shapefile...")
            QApplication.processEvents()
            
            chips_folder, results_folder, annotated_path = post_process(
                image_path=image_path,
                results_list=results_list,
                project_path=project_path,
                experiment_name=experiment_name,
                chip_height=chip_size,
                height_starts=height_starts,
                width_starts=width_starts
            )
            
            self.status_label.setText("Cleaning up temporary files...")
            QApplication.processEvents()
            
            for folder in [chips_folder, results_folder]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                        print(f"Deleted folder: {folder}")
                    except Exception as e:
                        print(f"Failed to delete {folder}: {str(e)}")
            
            if os.path.exists(annotated_path):
                try:
                    os.remove(annotated_path)
                    print(f"Deleted annotated image: {annotated_path}")
                except Exception as e:
                    print(f"Failed to delete {annotated_path}: {str(e)}")
            
            if os.path.exists(chips_dir) and not os.listdir(chips_dir):
                try:
                    shutil.rmtree(chips_dir)
                    print(f"Deleted empty chips parent folder: {chips_dir}")
                except Exception as e:
                    print(f"Failed to delete {chips_dir}: {str(e)}")
            
            self.status_label.setText("Completed! Check project folder for shapefile")
            QMessageBox.information(self, "Success", "Inference, shapefile export, and cleanup completed.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")
            self.status_label.setText("Error occurred")
        
        finally:
            self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())