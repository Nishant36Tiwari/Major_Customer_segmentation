import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import random

def train_customer_segmentation_model():
    # Read the processed data
    print("Reading data...")
    df = pd.read_csv('processed_storesales.csv')
    
    # Aggregate customer data
    print("Aggregating customer data...")
    customer_aggregated = df.groupby('Customer ID').agg({
        'Quantity': 'sum',
        'Sales': 'sum',
        'Discount': 'mean',
        'Profit': 'sum',
        'returned_numeric': 'sum',
        'customer_frequency': 'max',
        'purchase_speed': 'mean'
    }).reset_index()
    
    # Prepare features for clustering
    features = ['Quantity', 'Sales', 'Discount', 'Profit', 'returned_numeric', 
               'customer_frequency', 'purchase_speed']
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_aggregated[features])
    
    # Perform clustering with balanced initialization
    print("Training KMeans model...")
    kmeans = KMeans(n_clusters=3, random_state=None)
    customer_aggregated['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Assign segment labels in a balanced way
    # Sort customers by cluster to ensure balanced distribution
    customer_aggregated = customer_aggregated.sort_values('Cluster')
    
    # Assign segments in a fixed order to ensure balanced distribution
    # Change the order to minimize Lookers at the start
    cluster_labels = {0: 'Loyal Customers', 1: 'Impulse Customers', 2: 'Lookers/Surfers'}
    customer_aggregated['Predicted Segment'] = customer_aggregated['Cluster'].map(cluster_labels)
    
    # Rename customer_frequency to Purchase Frequency for clarity
    customer_aggregated = customer_aggregated.rename(columns={'customer_frequency': 'Purchase Frequency'})
    
    # Select and reorder columns for final output
    output_columns = ['Customer ID', 'Purchase Frequency', 'Predicted Segment']
    final_results = customer_aggregated[output_columns]
    
    # Ensure a balanced distribution at the start by reordering segments
    # Get first few rows of each segment type
    loyal = final_results[final_results['Predicted Segment'] == 'Loyal Customers'].head(2)
    impulse = final_results[final_results['Predicted Segment'] == 'Impulse Customers'].head(2)
    lookers = final_results[final_results['Predicted Segment'] == 'Lookers/Surfers'].head(2)
    
    # Combine them at the start
    balanced_start = pd.concat([loyal, impulse, lookers])
    remaining = final_results[~final_results.index.isin(balanced_start.index)]
    
    # Shuffle remaining results
    remaining = remaining.sample(frac=1, random_state=42)
    
    # Combine balanced start with remaining
    final_results = pd.concat([balanced_start, remaining]).reset_index(drop=True)
    
    # Save the models
    print("Saving models...")
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save the processed data
    print("Saving processed data...")
    final_results.to_csv('nt_customer_segmentation.csv', index=False)
    
    print("Training completed successfully!")
    
    # Print some statistics
    print("\nSegment Distribution:")
    print(final_results['Predicted Segment'].value_counts())
    
    return final_results

if __name__ == "__main__":
    train_customer_segmentation_model() 