import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Customer Journey Analysis Using Clustering and Dimensionality Reduction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Customer Behavior CSV", type=["csv"])

if uploaded_file is not None:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset:")
    st.write(df.head())

    # Select numeric columns only for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)

    if len(selected_features) < 2:
        st.warning("Please select at least two features.")
    else:
        # Drop rows with missing values in selected features
        data = df[selected_features].dropna()

        # Keep only rows with no missing values in main df
        df = df.loc[data.index]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Choose number of clusters
        n_clusters = st.slider("Select number of clusters", 2, 10, 4)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # UMAP Dimensionality Reduction
        reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        embedding = reducer.fit_transform(scaled_data)
        df['UMAP_1'] = embedding[:, 0]
        df['UMAP_2'] = embedding[:, 1]

        # Cluster Visualization
        st.subheader("Cluster Visualization (UMAP)")
        fig, ax = plt.subplots()
        sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='Cluster', data=df, palette='tab10', ax=ax)
        plt.title("Customer Segments Visualized using UMAP")
        st.pyplot(fig)

        # Display Clustered Data
        st.subheader("Clustered Data")
        st.write(df.head())

        # Download clustered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", data=csv, file_name="clustered_customers.csv", mime='text/csv')

