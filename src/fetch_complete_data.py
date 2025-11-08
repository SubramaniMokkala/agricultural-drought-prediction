import ee
import pandas as pd
from datetime import datetime
import time

# Initialize Earth Engine
ee.Initialize(project='drought-analysis-2025')

# Define study region - Maharashtra
maharashtra = ee.Geometry.Rectangle([72.6, 15.6, 80.9, 22.0])

# Date range: 2015-2024
start_date = '2015-01-01'
end_date = '2024-10-31'

print("=" * 60)
print("FETCHING COMPLETE DROUGHT DATASET (2015-2024)")
print("=" * 60)
print(f"Region: Maharashtra, India")
print(f"Period: {start_date} to {end_date}")
print("=" * 60)

# Function to extract monthly data
def get_monthly_data(year, month):
    """Extract all features for a given month"""
    
    # Date range for the month
    month_start = f'{year}-{month:02d}-01'
    if month == 12:
        month_end = f'{year+1}-01-01'
    else:
        month_end = f'{year}-{month+1:02d}-01'
    
    try:
        # 1. NDVI (Vegetation Health) - MODIS
        ndvi = ee.ImageCollection('MODIS/061/MOD13A2') \
            .filterDate(month_start, month_end) \
            .filterBounds(maharashtra) \
            .select('NDVI') \
            .mean() \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=maharashtra,
                scale=1000
            ).getInfo()
        
        ndvi_value = ndvi.get('NDVI', None)
        if ndvi_value:
            ndvi_value = ndvi_value / 10000  # Scale factor
        
        # 2. Precipitation - CHIRPS
        precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(month_start, month_end) \
            .filterBounds(maharashtra) \
            .select('precipitation') \
            .sum() \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=maharashtra,
                scale=5000
            ).getInfo()
        
        precip_value = precip.get('precipitation', None)
        
        # 3. Temperature - ERA5
        temp = ee.ImageCollection('ECMWF/ERA5/DAILY') \
            .filterDate(month_start, month_end) \
            .filterBounds(maharashtra) \
            .select(['mean_2m_air_temperature', 'maximum_2m_air_temperature', 'minimum_2m_air_temperature']) \
            .mean() \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=maharashtra,
                scale=10000
            ).getInfo()
        
        temp_mean = temp.get('mean_2m_air_temperature', None)
        temp_max = temp.get('maximum_2m_air_temperature', None)
        temp_min = temp.get('minimum_2m_air_temperature', None)
        
        # Convert from Kelvin to Celsius
        if temp_mean:
            temp_mean = temp_mean - 273.15
        if temp_max:
            temp_max = temp_max - 273.15
        if temp_min:
            temp_min = temp_min - 273.15
        
        return {
            'year': year,
            'month': month,
            'date': f'{year}-{month:02d}',
            'ndvi': round(ndvi_value, 4) if ndvi_value else None,
            'precipitation_mm': round(precip_value, 2) if precip_value else None,
            'temp_mean_c': round(temp_mean, 2) if temp_mean else None,
            'temp_max_c': round(temp_max, 2) if temp_max else None,
            'temp_min_c': round(temp_min, 2) if temp_min else None
        }
    
    except Exception as e:
        print(f"  Error for {year}-{month:02d}: {str(e)}")
        return None

# Collect data for all months
all_data = []
total_months = 0

for year in range(2015, 2025):
    end_month = 10 if year == 2024 else 12  # Only up to October 2024
    
    print(f"\nProcessing Year: {year}")
    print("-" * 60)
    
    for month in range(1, end_month + 1):
        print(f"  Fetching data for {year}-{month:02d}...", end=" ")
        
        data = get_monthly_data(year, month)
        
        if data:
            all_data.append(data)
            print(f"✓ NDVI: {data['ndvi']}, Precip: {data['precipitation_mm']}mm")
            total_months += 1
        else:
            print("✗ Failed")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
df.to_csv('data/drought_dataset_2015_2024.csv', index=False)

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE!")
print("=" * 60)
print(f"Total months collected: {total_months}")
print(f"Dataset shape: {df.shape}")
print(f"Saved to: data/drought_dataset_2015_2024.csv")
print("\nFirst few rows:")
print(df.head(10))
print("\nDataset info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())
print("=" * 60)