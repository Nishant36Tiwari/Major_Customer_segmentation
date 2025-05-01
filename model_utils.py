from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import joblib

def train_model():
    df = pd.read_csv('processed_storesales.csv')  # ya jo bhi file use kar rahe ho
    features = df.drop(['CustomerID'], axis=1)  # Adjust as per your file

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Save PCA model
    joblib.dump(pca, 'pca_model.pkl')

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_pca)

    # Save KMeans model
    joblib.dump(kmeans, 'kmeans_model.pkl')

    print("✅ Models trained & saved: scaler.pkl, pca_model.pkl, kmeans_model.pkl")
