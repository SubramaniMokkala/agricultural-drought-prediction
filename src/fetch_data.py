import ee
import pandas as pd
from datetime import datetime
import json

# Initialize Earth Engine
ee.Initialize(project='drought-analysis-2025')

# Define study region - Maharashtra
maharashtra = ee.Geometry.Rectangle([72.6, 15.6, 80.9, 22.0])

# Date range
start_date = '2020-01-01'
end_date = '2024-12-31'

print("Fetching NDVI data (Vegetation Health)...")
print("=" * 50)

# Get MODIS NDVI data
ndvi_collection = ee.ImageCollection('MODIS/006/MOD13A2') \
    .filterDate(start_date, end_date) \
    .filterBounds(maharashtra) \
    .select('NDVI')

# Get collection info
count = ndvi_collection.size().getInfo()
print(f"Total NDVI images available: {count}")

# Sample one image to check
first_image = ndvi_collection.first()
date_info = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print(f"First image date: {date_info}")

print("\n" + "=" * 50)
print("Fetching Precipitation data...")
print("=" * 50)

# Get CHIRPS Precipitation data
precip_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterDate(start_date, end_date) \
    .filterBounds(maharashtra) \
    .select('precipitation')

count_precip = precip_collection.size().getInfo()
print(f"Total precipitation images available: {count_precip}")

# Calculate monthly precipitation for 2023
precip_2023 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterDate('2023-01-01', '2023-12-31') \
    .filterBounds(maharashtra) \
    .select('precipitation')

# Aggregate by month
monthly_precip = []
for month in range(1, 13):
    month_data = precip_2023.filter(ee.Filter.calendarRange(month, month, 'month')) \
        .sum() \
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=maharashtra,
            scale=5000
        ).getInfo()
    
    monthly_precip.append({
        'month': month,
        'total_precipitation_mm': round(month_data['precipitation'], 2)
    })
    print(f"Month {month}: {round(month_data['precipitation'], 2)} mm")

# Save to CSV
df = pd.DataFrame(monthly_precip)
df.to_csv('data/monthly_precipitation_2023.csv', index=False)

print("\n" + "=" * 50)
print("✓ Data fetched successfully!")
print("✓ Saved: data/monthly_precipitation_2023.csv")
print("=" * 50)