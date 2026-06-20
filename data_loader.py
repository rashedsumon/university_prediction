import os
import glob
import pandas as pd
import kagglehub
import streamlit as st

def load_university_data():
    """
    Downloads the Wikipedia University dataset and automatically cleans column names
    to fix KeyErrors caused by unexpected casing or white spaces.
    """
    try:
        # Check if Kaggle credentials exist in Streamlit Secrets
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
        
        # Download dataset
        path = kagglehub.dataset_download("muhamamdnaveed/wikipedia-datasets-of-universities")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded dataset.")
            
        df = pd.read_csv(csv_files[0])
        
        # --- FIX KEYERROR: Clean Column Names ---
        # Strip trailing/leading spaces and make a mapping dictionary
        df.columns = df.columns.str.strip()
        col_map = {col.lower(): col for col in df.columns}
        
        # Target columns we need, matched case-insensitively
        target_university = col_map.get('universitylabel', col_map.get('university', df.columns[0]))
        target_country = col_map.get('countrylabel', col_map.get('country', df.columns[1] if len(df.columns) > 1 else df.columns[0]))
        target_coord = col_map.get('coord', col_map.get('coordinates', None))
        target_inception = col_map.get('inception', col_map.get('founded', col_map.get('year', None)))

        # Rename columns uniformly so model.py and streamlit_app.py don't break
        rename_dict = {
            target_university: 'universityLabel',
            target_country: 'countryLabel'
        }
        if target_coord:
            rename_dict[target_coord] = 'coord'
        if target_inception:
            rename_dict[target_inception] = 'inception'
            
        df = df.rename(columns=rename_dict)

        # Double check core columns exist now, if not create default empty ones
        if 'universityLabel' not in df.columns:
            df['universityLabel'] = "Unknown University"
        if 'countryLabel' not in df.columns:
            df['countryLabel'] = "Unknown Country"
        if 'inception' not in df.columns:
            df['inception'] = "N/A"

        # Safe Coordinate Extraction
        if 'coord' in df.columns:
            df['coord'] = df['coord'].astype(str)
            def extract_lon(x):
                try: return float(x.split('(')[-1].split(')')[0].split()[0]) if 'Point' in x else None
                except: return None
            def extract_lat(x):
                try: return float(x.split('(')[-1].split(')')[0].split()[1]) if 'Point' in x else None
                except: return None
                
            df['lon'] = df['coord'].apply(extract_lon)
            df['lat'] = df['coord'].apply(extract_lat)
        
        # Final cleanup of nulls
        df['universityLabel'] = df['universityLabel'].fillna("Unknown University")
        df['countryLabel'] = df['countryLabel'].fillna("Unknown Country")
        df['inception'] = df['inception'].fillna("N/A")
        
        return df

    except Exception as e:
        st.sidebar.error(f"KaggleHub Data Parsing Error: {e}")
        return pd.DataFrame()