import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load processed data
df = pd.read_csv('processed_storesales.csv')

# Drop non-numeric columns
columns_to_drop = ['Customer ID', 'Customer Name', 'Segment', 'Category', 'Region']
df = df.drop(columns=columns_to_drop)

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Apply PCA (2D)
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

# Save PCA model
joblib.dump(pca, 'pca_model.pkl')
print("✅ PCA model saved as pca_model.pkl")
