import ee
import pandas as pd
from datetime import datetime

# Initialize Earth Engine
print("Initializing Google Earth Engine...")
ee.Initialize(project='drought-analysis-2025')
print("✓ Earth Engine initialized successfully!\n")

# Define study region - Maharashtra, India
# Coordinates: [longitude, latitude] for bounding box
maharashtra = ee.Geometry.Rectangle([72.6, 15.6, 80.9, 22.0])

print("Study Region: Maharashtra, India")
print("Time Period: 2010-2025\n")

# Available datasets we'll use
print("=" * 50)
print("AVAILABLE DATASETS")
print("=" * 50)

datasets = {
    "MODIS NDVI (Vegetation Health)": "MODIS/006/MOD13A2",
    "CHIRPS Precipitation": "UCSB-CHG/CHIRPS/DAILY",
    "ERA5 Temperature": "ECMWF/ERA5/DAILY",
    "Soil Moisture": "NASA_USDA/HSL/SMAP10KM_soil_moisture"
}

for name, dataset_id in datasets.items():
    print(f"\n{name}")
    print(f"  Dataset ID: {dataset_id}")
    
print("\n" + "=" * 50)
print("Setup complete! Ready to fetch data.")
print("=" * 50)