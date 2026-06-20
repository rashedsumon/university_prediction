import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize a lightweight, high-performance AI text embedding model
# We cache this resource so it only loads once into memory on Streamlit Cloud
import streamlit as st

@st.cache_resource
def load_embedding_model():
    """Loads and caches the NLP model to compute sentence embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def prepare_search_index(df):
    """
    Creates a 'search text' profile for each university by combining 
    its name, country, and inception year.
    """
    # Create a descriptive text chunk for the AI to read
    df['search_profile'] = df.apply(
        lambda row: f"University Name: {row['universityLabel']}. Country: {row['countryLabel']}. Founded/Inception: {str(row['inception'])}.", 
        axis=1
    )
    return df

def perform_semantic_search(query, df, model, top_n=5):
    """
    Compares the user's question against the dataset using cosine similarity 
    and returns the top matches.
    """
    if df.empty or not query:
        return pd.DataFrame()
        
    # Ensure the search profiles exist
    if 'search_profile' not in df.columns:
        df = prepare_search_index(df)
        
    # 1. Convert university text data into AI embeddings
    dataset_embeddings = model.encode(df['search_profile'].tolist(), show_progress_bar=False)
    
    # 2. Convert user's question into an AI embedding
    query_embedding = model.encode([query])
    
    # 3. Calculate how close the question's meaning is to each university row
    similarity_scores = cosine_similarity(query_embedding, dataset_embeddings)[0]
    
    # 4. Attach scores and sort by most relevant
    df_copy = df.copy()
    df_copy['AI_Relevance_Score'] = similarity_scores
    
    # Filter to return only the best matches
    results = df_copy.sort_values(by='AI_Relevance_Score', ascending=False).head(top_n)
    return results