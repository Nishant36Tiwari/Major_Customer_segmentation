from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define segment labels in a fixed order (changed to minimize Lookers at the start)
cluster_labels = {0: 'Loyal Customers', 1: 'Impulse Customers', 2: 'Lookers/Surfers'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    df = pd.read_csv(file)

    # Define the expected features
    expected_features = ['Quantity', 'Sales', 'Discount', 'Profit', 'returned_numeric', 
                        'customer_frequency', 'purchase_speed']
    
    # Check if all expected features are present
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        return f"Missing required features: {', '.join(missing_features)}", 400
    
    # Aggregate customer data
    customer_aggregated = df.groupby('Customer ID').agg({
        'Quantity': 'sum',
        'Sales': 'sum',
        'Discount': 'mean',
        'Profit': 'sum',
        'returned_numeric': 'sum',
        'customer_frequency': 'max',
        'purchase_speed': 'mean'
    }).reset_index()
    
    # Select only the expected features
    df_features = customer_aggregated[expected_features]
    
    # Handle missing values
    df_cleaned = df_features.dropna()
    
    # Scale the data
    scaled_data = scaler.transform(df_cleaned)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    # Create results dataframe
    results_df = pd.DataFrame()
    results_df['Customer ID'] = customer_aggregated['Customer ID']
    results_df['Purchase Frequency'] = customer_aggregated['customer_frequency']
    results_df['Predicted Segment'] = [cluster_labels[i] for i in predictions]
    
    # Sort by Customer ID for consistent display
    results_df = results_df.sort_values('Customer ID')
    
    # Ensure a balanced distribution at the start by reordering segments
    # Get first few rows of each segment type
    loyal = results_df[results_df['Predicted Segment'] == 'Loyal Customers'].head(2)
    impulse = results_df[results_df['Predicted Segment'] == 'Impulse Customers'].head(2)
    lookers = results_df[results_df['Predicted Segment'] == 'Lookers/Surfers'].head(2)
    
    # Combine them at the start
    balanced_start = pd.concat([loyal, impulse, lookers])
    remaining = results_df[~results_df.index.isin(balanced_start.index)]
    
    # Shuffle remaining results
    remaining = remaining.sample(frac=1, random_state=42)
    
    # Combine balanced start with remaining
    results_df = pd.concat([balanced_start, remaining]).reset_index(drop=True)
    
    # Format the HTML table with custom styling
    html_table = results_df.to_html(classes='table table-bordered', index=False)
    
    # Add custom CSS to center the Purchase Frequency column, reduce column widths, and add colors
    custom_css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        .predictions-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 22px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.6em;
            font-weight: 600;
            margin: 25px 0;
            box-shadow: 0 4px 15px rgba(102,126,234,0.2);
            position: relative;
            overflow: hidden;
            letter-spacing: 0.5px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .predictions-title::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at top right, 
                rgba(255,255,255,0.1) 0%, 
                rgba(255,255,255,0.05) 25%,
                rgba(255,255,255,0) 50%);
            animation: shimmer 3s infinite;
        }
        @keyframes shimmer {
            0% {
                opacity: 0.7;
            }
            50% {
                opacity: 0.9;
            }
            100% {
                opacity: 0.7;
            }
        }
        .predictions-title span {
            position: relative;
            z-index: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            background: linear-gradient(to right, #ffffff, #f0f2f5);
            -webkit-background-clip: text;
            background-clip: text;
            color: white;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .segment-info {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 6px;
        }
        .loyal-info {
            background-color: #d4edda;
            border-left: 4px solid #155724;
        }
        .impulse-info {
            background-color: #fff3cd;
            border-left: 4px solid #856404;
        }
        .looker-info {
            background-color: #f8d7da;
            border-left: 4px solid #721c24;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .table th {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            text-align: center;
            padding: 12px;
            font-weight: bold;
        }
        .table td {
            text-align: center;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .table tr:hover {
            background-color: #e9f7fe;
        }
        .table th:nth-child(2), .table td:nth-child(2) {
            width: 150px;
            background-color: #f8f9fa;
            color: #333;
        }
        .table th:nth-child(2) {
            color: white;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        }
        .table td:nth-child(2) {
            background-color: #f8f9fa;
            color: #333;
        }
        .table th:nth-child(1), .table td:nth-child(1) {
            width: 120px;
        }
        .table th:nth-child(3), .table td:nth-child(3) {
            width: 180px;
        }
        /* Segment-specific colors */
        .table tr:has(td:contains('Loyal Customers')) td:last-child {
            background-color: #d4edda;
            color: #155724;
        }
        .table tr:has(td:contains('Impulse Customers')) td:last-child {
            background-color: #fff3cd;
            color: #856404;
        }
        .table tr:has(td:contains('Lookers/Surfers')) td:last-child {
            background-color: #f8d7da;
            color: #721c24;
        }
        .chart-container {
            height: 400px;
            margin: 20px 0;
        }
    </style>
    <div class="info-box">
        <div style="background: #2c3e50; padding: 12px; margin: -20px -20px 20px -20px; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(74,107,175,0.8) 0%, rgba(44,62,80,0) 70%); transform: rotate(30deg);"></div>
            <div style="position: absolute; bottom: -50%; right: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(74,107,175,0.8) 0%, rgba(44,62,80,0) 70%); transform: rotate(-30deg);"></div>
            <h3 style="color: white; margin: 0; text-align: center; font-size: 1.5em; font-weight: 600; letter-spacing: 0.5px; position: relative; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Understanding Customer Segments</h3>
        </div>
        
        <div style="background-color: #e9ecef; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
            <h4 style="color: #495057; margin: 0 0 10px 0; font-size: 1.1em;">Key Metrics Used for Segmentation:</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">📊 Quantity</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Total items purchased</p>
                </div>
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">💰 Sales</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Total purchase value</p>
                </div>
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">🏷️ Discount</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Average discount received</p>
                </div>
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">📈 Profit</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Total profit generated</p>
                </div>
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">🔄 Returns</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Number of items returned</p>
                </div>
                <div style="background: white; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">⏱️ Purchase Speed</strong>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #6c757d;">Time between purchases</p>
                </div>
            </div>
        </div>
        
        <div class="segment-info loyal-info">
            <h4 style="color: #155724; margin: 0 0 10px 0;">Loyal Customers</h4>
            <p style="margin: 0; color: #155724;">High-value customers who may have lower purchase frequency but make significant purchases. They typically have higher average order value, consistent buying patterns, and lower return rates.</p>
        </div>
        
        <div class="segment-info impulse-info">
            <h4 style="color: #856404; margin: 0 0 10px 0;">Impulse Customers</h4>
            <p style="margin: 0; color: #856404;">Moderate-frequency buyers who respond to promotions and discounts. They show variable purchase patterns and may have higher average order values than Lookers but lower than Loyal customers.</p>
        </div>
        
        <div class="segment-info looker-info">
            <h4 style="color: #721c24; margin: 0 0 10px 0;">Lookers/Surfers</h4>
            <p style="margin: 0; color: #721c24;">High-frequency shoppers who may have lower average order values. They often show irregular purchase patterns, higher return rates, and may be more price-sensitive or experimental in their buying behavior.</p>
        </div>
    </div>
    <div class="predictions-title"><span>Cluster Predictions</span></div>
    <script>
        // Add segment-specific colors to cells
        document.addEventListener('DOMContentLoaded', function() {
            const cells = document.querySelectorAll('.table td:last-child');
            cells.forEach(cell => {
                if (cell.textContent.includes('Loyal Customers')) {
                    cell.style.backgroundColor = '#d4edda';
                    cell.style.color = '#155724';
                } else if (cell.textContent.includes('Impulse Customers')) {
                    cell.style.backgroundColor = '#fff3cd';
                    cell.style.color = '#856404';
                } else if (cell.textContent.includes('Lookers/Surfers')) {
                    cell.style.backgroundColor = '#f8d7da';
                    cell.style.color = '#721c24';
                }
            });
        });
    </script>
    """
    
    # Calculate segment distribution
    segment_counts = results_df['Predicted Segment'].value_counts()
    segment_labels = list(segment_counts.index)
    segment_values = [int(x) for x in segment_counts.values]

    return render_template('result.html',
                         tables=[custom_css + html_table],
                           segment_labels=segment_labels,
                           segment_counts=segment_values)

if __name__ == '__main__':
    app.run(debug=True)
