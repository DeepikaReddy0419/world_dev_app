
# world_dev_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Streamlit setup
st.set_page_config(page_title="🌍 World Development - EDA & Clustering", layout="wide")
st.title("🌍 World Development Analysis & Clustering")

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigation",
    ["Upload & Preview", "Data Cleaning", "Univariate Analysis", "Bivariate Analysis",
     "Outlier Handling", "Correlation & Plots", "PCA & Clustering"]
)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file, engine='openpyxl')

    # -------------------------------
    # Section 1: Upload & Preview
    # -------------------------------
    if menu == "Upload & Preview":
        st.subheader("📊 Raw Data Preview")
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

    # -------------------------------
    # Section 2: Data Cleaning
    # -------------------------------
    if menu == "Data Cleaning":
        st.subheader("🧹 Data Cleaning")

        # Currency cleaning
        currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Business Tax Rate
        if "Business Tax Rate" in df.columns:
            df['Business Tax Rate'] = df['Business Tax Rate'].astype(str).str.replace('%', '')
            df['Business Tax Rate'] = pd.to_numeric(df['Business Tax Rate'], errors='coerce')

        # Fill missing values
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        st.write("✅ Missing values filled with mean for numerical columns")
        st.dataframe(df.head())

    # -------------------------------
    # Section 3: Univariate Analysis
    # -------------------------------
    if menu == "Univariate Analysis":
        st.subheader("📈 Histograms & Boxplots")

        numerical_cols = df.select_dtypes(include=np.number).columns

        for col in numerical_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], bins=50, kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

    # -------------------------------
    # Section 4: Bivariate Analysis
    # -------------------------------
    if menu == "Bivariate Analysis":
        st.subheader("🔀 Bivariate Analysis")

        if "Birth Rate" in df.columns and "Infant Mortality Rate" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x="Birth Rate", y="Infant Mortality Rate", ax=ax)
            ax.set_title("Birth Rate vs Infant Mortality Rate")
            st.pyplot(fig)

        if "Life Expectancy Female" in df.columns and "Life Expectancy Male" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x="Life Expectancy Female", y="Life Expectancy Male", ax=ax)
            ax.set_title("Life Expectancy Female vs Male")
            st.pyplot(fig)

    # -------------------------------
    # Section 5: Outlier Handling
    # -------------------------------
    if menu == "Outlier Handling":
        st.subheader("📉 Outlier Detection & Replacement")
        numerical_cols = df.select_dtypes(include=np.number).columns

        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            lower = q1 - 1.5 * IQR
            upper = q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]

            if not outliers.empty:
                st.write(f"Outliers in {col}: {len(outliers)} replaced with mean")
                df.loc[outliers.index, col] = df[col].mean()

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col} (After Outlier Handling)")
            st.pyplot(fig)

    # -------------------------------
    # Section 6: Correlation & Plots
    # -------------------------------
    if menu == "Correlation & Plots":
        st.subheader("📊 Correlation Heatmap")
        numerical_cols = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Section 7: PCA & Clustering
    # -------------------------------
    if menu == "PCA & Clustering":
        st.subheader("🔎 PCA - Dimensionality Reduction")

        numerical_df = df.select_dtypes(include=np.number)

        # Replace inf/-inf with NaN, then fill with column means
        numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
        numerical_df = numerical_df.fillna(numerical_df.mean())

        # Only keep columns that are fully numeric now
        numerical_df = numerical_df.loc[:, numerical_df.notnull().all()]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_df)


        pca = PCA()
        pca.fit(scaled_data)

        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        st.pyplot(fig)

        # 3D Scatter Plot
        if scaled_data.shape[1] >= 3:
            pcs = pca.transform(scaled_data)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], alpha=0.6)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("3D PCA Scatter Plot")
            st.pyplot(fig)
