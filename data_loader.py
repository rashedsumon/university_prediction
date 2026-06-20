import os
import glob
import pandas as pd
import kagglehub

def load_university_data():
    """
    Automatically downloads the latest version of the Wikipedia 
    University dataset using kagglehub and loads it into a Pandas DataFrame.
    """
    try:
        # Download latest version of the dataset
        path = kagglehub.dataset_download("muhamamdnaveed/wikipedia-datasets-of-universities")
        
        # Search for any CSV file inside the downloaded path
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded dataset.")
            
        # Read the first available CSV file
        df = pd.read_csv(csv_files[0])
        
        # Basic Data Cleaning & Feature Extraction
        # Parse coordinates 'Point(lon lat)' if present
        if 'coord' in df.columns:
            df['coord'] = df['coord'].astype(str)
            # Extract longitude and latitude safely
            df['lon'] = df['coord'].apply(lambda x: float(x.split('(')[-1].split(')')[0].split()[0]) if 'Point' in x else None)
            df['lat'] = df['coord'].apply(lambda x: float(x.split('(')[-1].split(')')[0].split()[1]) if 'Point' in x else None)
        
        # Fill missing values for cleaner rendering
        df['universityLabel'] = df['universityLabel'].fillna("Unknown University")
        df['countryLabel'] = df['countryLabel'].fillna("Unknown Country")
        
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure