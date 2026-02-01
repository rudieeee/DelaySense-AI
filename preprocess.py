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
    # Generate realistic sample data for testing
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
# Date conversions (handle errors)
df['Order_Date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['Shipping_Date'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')  
# Note: Dataset doesn't have delivery due/actual dates, we'll estimate them
df['Delivery_Due'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipment (scheduled)'], unit='D')
df['Delivery_Actual_Date'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipping (real)'], unit='D')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius km
    dlat, dlon = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# Distance from pickup to drop (fill NaN with median)
# Note: Dataset only has one lat/lon pair, so we'll use customer and order locations
df['pickup_lat'] = df['Latitude'].fillna(df['Latitude'].median())
df['pickup_lon'] = df['Longitude'].fillna(df['Longitude'].median())
# Use a random offset to simulate delivery location
np.random.seed(42)
df['drop_lat'] = df['pickup_lat'] + np.random.uniform(-5, 5, len(df))
df['drop_lon'] = df['pickup_lon'] + np.random.uniform(-5, 5, len(df))
df['distance_km'] = haversine(df['pickup_lat'], df['pickup_lon'], df['drop_lat'], df['drop_lon'])

# Times
df['processing_time_days'] = (df['Shipping_Date'] - df['Order_Date']).dt.days.fillna(df['Days for shipping (real)'].fillna(1))
df['scheduled_days'] = df['Days for shipment (scheduled)'].fillna(5)
df['actual_days'] = df['Days for shipping (real)'].fillna(5)
df['delay_days'] = df['actual_days'] - df['scheduled_days']

# Volume & risk
df['order_volume'] = df['Order Item Quantity'].fillna(1) * (1 - df['Order Item Discount Rate'].fillna(0).abs()) + 1  # Proxy
df['risk_score'] = np.clip(df['delay_days'].rolling(100, min_periods=1).mean().fillna(1.0), 0.5, 2.0)

# External factors (simulate realistic; replace with API)
np.random.seed(42)
df['weather_rain'] = np.random.binomial(1, 0.15 + 0.1*(df['delay_days']>0), len(df))
df['peak_traffic'] = df['Order_Date'].dt.hour.fillna(12).isin([7,8,17,18,19]).astype(int)

# Multi-class: 0=On-Time (delay<=0), 1=At Risk (0<delay<=3), 2=Delayed (>3)
df['risk_level'] = np.where(df['delay_days'] <= 0, 0,
                  np.where(df['delay_days'] <= 3, 1, 2))

# Clean & select (150k rows)
feat_cols = ['processing_time_days', 'scheduled_days', 'actual_days', 'delay_days', 
             'distance_km', 'order_volume', 'risk_score', 'weather_rain', 'peak_traffic']
df_clean = df[feat_cols + ['risk_level']].dropna()
print(f"Dataset ready: {len(df_clean)} rows")
print("Class dist:", df_clean['risk_level'].value_counts(normalize=True).round(3))

df_clean.to_csv('delivery_data.csv', index=False)
print("✅ delivery_data.csv saved. Run: python train.py")
