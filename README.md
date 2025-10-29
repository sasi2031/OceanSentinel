# OceanSentinel

**Automated desktop tool for running YOLO-OBB inference on large GeoTIFF satellite imagery and exporting georeferenced shapefiles.**

---

## ✨ Overview & Core Features

OceanSentinel is designed to bridge the gap between deep learning models and geospatial analysis. It streamlines the entire workflow required to detect objects in large satellite scenes, specifically targeting ships, and outputs results compatible with any standard GIS software.

### Key Features:

* **GeoTIFF Processing:** Directly processes large GeoTIFFs (supporting both SAR and Optical data).
* **Intelligent Chipping:** Automatically clips images into smaller, overlapping chips (e.g., 1024x1024) to ensure robust detection, especially near tile boundaries.
* **YOLO-OBB Inference:** Integrates with YOLOv8 to perform Oriented Bounding Box (OBB) detection.
* **Geospatial Export:** Stitches detections back to the original image coordinates and exports a final **georeferenced Shapefile (.shp)**, including confidence scores and class names.
* **GUI Interface:** Easy-to-use desktop application built with PyQt6.
* **Cleanup:** Automatically deletes all temporary chip files and intermediate YOLO result folders after a successful run.

---

## 📥 Getting Started

There are two ways to use ChipperV2. We strongly recommend the Windows Executable for simplicity.

### 🚀 Option 1: Windows Standalone Executable (Recommended)

This method requires **no Python installation** or dependency management.

1.  **Download:** Get the latest Windows executable (`OceanSentinel.exe`) for the desired version from this external link:
    * ➡️ **[Download .exe](https://azistaindustries-my.sharepoint.com/:f:/p/sasikanth_jada/EgMTITfJimtBoRlCf376KdwB_RdvkfWysMdQZpxlWtYPBg?e=8oqo8j)**
2.  **Run:** Double-click the downloaded `.exe` file to start the application.
3.  **Next:** Proceed to the **Model Prerequisites** section below.

### 🛠️ Option 2: Run from Source

1.  **Clone/Download:** Download the `ChipperV2.py` script from this repository.
2.  **Install Python:** Ensure you have Python 3.8+ installed.
3.  **Install Dependencies:** Install the required libraries:
    ```bash
    pip install numpy matplotlib rasterio ultralytics geopandas shapely opencv-python PyQt6
    ```
4.  **Run:** Execute the script:
    ```bash
    python ChipperV2.py
    ```

---

## ⚠️ Model Prerequisites (Crucial Step)

The application requires pre-trained model weights (`best.pt`) for both Sentinel-1 and Sentinel-2. **These files are too large for GitHub and are hosted externally.**

1.  **Download Models:** Download the complete `models` package from the following link:
    * ➡️ **[Download Model Weights Package Here](https://azistaindustries-my.sharepoint.com/:f:/p/sasikanth_jada/Eiik7y4rC_9GjJ51a11KkhkBJs4V63RK7GiIUYbTZo-sTg?e=Vwbeb4)**
2.  **Structure Setup:** Place the downloaded **`models`** folder directly in the same directory as your `ChipperV2.py` script (or next to your executable).

Your final directory structure **must** look like this for the application to find the weights:
```
OceanSentinel/
├── ChipperV2.py
├── OceanSentinel.exe (if using the Windows build)
└── models/
    ├── sentinel1/
    │   └── best.pt  (Sentinel-1 Model Weights)
    └── sentinel2/
        └── best.pt  (Sentinel-2 Model Weights)
```
---

## 🔐 Rights & Wrongs

🕶️ Unauthorized duplication will be met with raised eyebrows

