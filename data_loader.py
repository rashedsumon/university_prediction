import os
import glob
import pandas as pd
import kagglehub
import streamlit as st

def load_university_data():
    """
    Automatically sets up Kaggle credentials and downloads the latest 
    version of the Wikipedia University dataset.
    """
    try:
        # Check if Kaggle credentials exist in Streamlit Secrets (for Cloud Deployment)
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
        
        # Download latest version of the dataset
        path = kagglehub.dataset_download("muhamamdnaveed/wikipedia-datasets-of-universities")
        
        # Search for any CSV file inside the downloaded path
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded dataset.")
            
        # Read the first available CSV file
        df = pd.read_csv(csv_files[0])
        
        # Basic Data Cleaning & Feature Extraction
        if 'coord' in df.columns:
            df['coord'] = df['coord'].astype(str)
            df['lon'] = df['coord'].apply(lambda x: float(x.split('(')[-1].split(')')[0].split()[0]) if 'Point' in x else None)
            df['lat'] = df['coord'].apply(lambda x: float(x.split('(')[-1].split(')')[0].split()[1]) if 'Point' in x else None)
        
        df['universityLabel'] = df['universityLabel'].fillna("Unknown University")
        df['countryLabel'] = df['countryLabel'].fillna("Unknown Country")
        
        return df

    except Exception as e:
        # Print the exact error log to the Streamlit console for easier debugging
        st.sidebar.error(f"KaggleHub Error Log: {e}")
        return pd.DataFrame()