import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Loading DataCo dataset...")
try:
    df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1', low_memory=False)
except FileNotFoundError:
    print("⚠️  DataCoSupplyChainDataset.csv not found. Generating sample data...")
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'Order Date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Shipping Date': pd.date_range('2023-01-01', periods=n_samples, freq='H') + pd.Timedelta(days=1),
        'Delivery Due Date': pd.date_range('2023-01-01', periods=n_samples, freq='H') + pd.Timedelta(days=5),
        'Delivery Actual Date': pd.date_range('2023-01-01', periods=n_samples, freq='H') + pd.Timedelta(days=5) + pd.to_timedelta(np.random.randint(-2, 8, n_samples), unit='D'),
        'Latitude': 40.7128 + np.random.randn(n_samples) * 5,
        'Longitude': -74.0060 + np.random.randn(n_samples) * 5,
        'Drop Latitude': 40.7128 + np.random.randn(n_samples) * 5,
        'Drop Longitude': -74.0060 + np.random.randn(n_samples) * 5,
        'Days for shipment (scheduled)': np.random.randint(3, 10, n_samples),
        'Days for shipping (real)': np.random.randint(2, 15, n_samples),
        'Order Item Quantity': np.random.randint(1, 20, n_samples),
        'Order Items Discount': np.random.uniform(0, 0.3, n_samples)
    })
    print(f"✅ Generated {len(df)} sample records")

print("Engineering features...")
# Date conversions 
df['Order_Date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['Shipping_Date'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')  

df['Delivery_Due'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipment (scheduled)'], unit='D')
df['Delivery_Actual_Date'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipping (real)'], unit='D')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat, dlon = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# Distance from pickup to drop 

df['pickup_lat'] = df['Latitude'].fillna(df['Latitude'].median())
df['pickup_lon'] = df['Longitude'].fillna(df['Longitude'].median())

np.random.seed(42)
df['drop_lat'] = df['pickup_lat'] + np.random.uniform(-5, 5, len(df))
df['drop_lon'] = df['pickup_lon'] + np.random.uniform(-5, 5, len(df))
df['distance_km'] = haversine(df['pickup_lat'], df['pickup_lon'], df['drop_lat'], df['drop_lon'])

# Times 
df['scheduled_days'] = df['Days for shipment (scheduled)'].fillna(5)

# Volume 
df['order_volume'] = df['Order Item Quantity'].fillna(1) * (1 - df['Order Item Discount Rate'].fillna(0).abs()) + 1

# Processing time (warehouse processing before shipment)
np.random.seed(42)
df['processing_time'] = np.random.uniform(0.5, 3.0, len(df))  # 0.5 to 3 days processing time

df['day_of_week'] = df['Order_Date'].dt.dayofweek  # 0=Monday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['Order_Date'].dt.month
df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)  # Nov-Dec peak
df['distance_category'] = pd.cut(df['distance_km'].fillna(df['distance_km'].median()), 
                                   bins=[0, 100, 500, 1000, 5000], 
                                   labels=[0,1,2,3]).astype(int)

# External factors 
np.random.seed(42)
df['weather_rain'] = np.random.binomial(1, 0.20, len(df))  # Random 20% chance of rain
df['peak_traffic'] = df['Order_Date'].dt.hour.fillna(12).isin([7,8,17,18,19]).astype(int)

np.random.seed(42)
base_delay = np.random.normal(0, 0.15, len(df))  

# Calculate speed requirement (km/day)
speed_required = df['distance_km'] / df['scheduled_days']

# Distance delay - realistic thresholds
# Very reasonable: < 150 km/day = minimal/no delay
# Tight: 150-300 km/day = moderate delay  
# Challenging: > 300 km/day = high delay
distance_delay = np.where(speed_required < 150, speed_required * 0.002,  
                 np.where(speed_required < 300, (speed_required - 100) / 60,
                         (speed_required - 100) / 35))

# Conditional delays - realistic impacts
weekend_delay = df['is_weekend'] * 1.2  # Weekends add 1.2 days
holiday_delay = df['is_holiday_season'] * 1.8  # Holidays add 1.8 days
weather_delay = df['weather_rain'] * 1.1  # Rain adds 1.1 day
traffic_delay = df['peak_traffic'] * 0.6  # Peak traffic adds 0.6 days
volume_delay = np.where(df['order_volume'] > 8, (df['order_volume'] - 8) * 0.15, 0)  # High volume adds delay
processing_delay = np.where(df['processing_time'] > 2.0, (df['processing_time'] - 2.0) * 0.25, 0)  # Only processing > 2 days

# Interaction effects - compounding when conditions align
interaction_delay = np.where(speed_required > 120, 
                            (df['is_weekend'] + df['weather_rain'] + df['peak_traffic']) * 0.35, 
                            0)

total_delay = base_delay + distance_delay + weekend_delay + holiday_delay + weather_delay + traffic_delay + volume_delay + processing_delay + interaction_delay
df['actual_days'] = df['scheduled_days'] + total_delay
df['delay_days'] = df['actual_days'] - df['scheduled_days']

# Multi-class with realistic thresholds:
# 0=On-Time (<= 1.5 days delay - accounts for normal variations)
# 1=At Risk (1.5-4 days delay - moderate concern)  
# 2=Delayed (> 4 days delay - significant problem)
df['risk_level'] = np.where(df['delay_days'] <= 1.5, 0,
                  np.where(df['delay_days'] <= 4, 1, 2))


feat_cols = ['scheduled_days', 'distance_km', 'order_volume', 'processing_time', 'weather_rain', 'peak_traffic',
             'day_of_week', 'is_weekend', 'month', 'is_holiday_season', 'distance_category']
df_clean = df[feat_cols + ['risk_level']].dropna()
print(f"Dataset ready: {len(df_clean)} rows")
print("Class dist:", df_clean['risk_level'].value_counts(normalize=True).round(3))
print(f"Features (no leakage): {feat_cols}")

df_clean.to_csv('delivery_data.csv', index=False)
print("✅ delivery_data.csv saved (no data leakage). Run: python train.py")
