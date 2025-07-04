import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_excel("data/Online Retail.xlsx", engine='openpyxl')

# Remove entries without CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Remove cancelled orders (InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Reference date for Recency calculation
ref_date = df['InvoiceDate'].max()

# RFM Calculation
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                              # Frequency
    'TotalPrice': 'sum'                                  # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Filter out customers with zero/negative monetary value
rfm = rfm[rfm['Monetary'] > 0]

# Save RFM scores
rfm.to_csv("rfm_scores.csv", index=False)
print("✅ RFM Scores saved to rfm_scores.csv")

# Clustering
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_features)

# Save clustered data
rfm.to_csv("rfm_clusters.csv", index=False)
print("✅ RFM Clustering completed. Output saved to rfm_clusters.csv")

