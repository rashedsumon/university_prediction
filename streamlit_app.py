import streamlit as fancy_ui
import pandas as pd
from data_loader import load_university_data
from model import load_embedding_model, perform_semantic_search

# Page Configuration
fancy_ui.set_page_config(page_title="AI University Search", layout="wide")

fancy_ui.title("🧠 University Prediction")
fancy_ui.write("Ask questions in natural language to find universities based on meaning, context, and history.")

# Step 1: Auto-Load Dataset via kagglehub
with fancy_ui.spinner("Downloading dataset from Kaggle Hub..."):
    raw_data = load_university_data()

# Step 2: Load the NLP AI Model
with fancy_ui.spinner("Loading AI Semantic Text Model..."):
    nlp_model = load_embedding_model()

if raw_data.empty:
    fancy_ui.error("Failed to load dataset. Please verify your internet connection or Kaggle settings.")
else:
    # Sidebar search tweaks
    fancy_ui.sidebar.header("🔍 Search Settings")
    max_results = fancy_ui.sidebar.slider("Maximum results to show", min_value=1, max_value=20, value=5)
    
    # User Input UI
    user_query = fancy_ui.text_input(
        "💬 Type your question or description below:", 
        placeholder="e.g., Which Ivy league universities are in the United States? / Give me historical universities from the 12th century."
    )
    
    if user_query:
        with fancy_ui.spinner("AI is analyzing your query and scanning historical data..."):
            # Execute the smart search
            search_results = perform_semantic_search(user_query, raw_data, nlp_model, top_n=max_results)
            
        if not search_results.empty:
            fancy_ui.subheader(f"✨ Top {max_results} AI-Ranked Matches")
            
            # Display results in a beautifully styled loop
            for idx, row in search_results.iterrows():
                # Convert match accuracy score to percentage format
                match_percentage = int(row['AI_Relevance_Score'] * 100)
                
                with fancy_ui.container():
                    # Create visual metric cards for each result
                    col1, col2 = fancy_ui.columns([4, 1])
                    with col1:
                        fancy_ui.markdown(f"### 🏛️ {row['universityLabel']}")
                        fancy_ui.markdown(f"**📍 Location:** {row['countryLabel']} | **📅 Founded/Inception:** {row['inception']}")
                    with col2:
                        
                    
                    fancy_ui.markdown("---")
                    
            # Bonus map feature if geographic data is available in the dataset
            if 'lat' in search_results.columns and 'lon' in search_results.columns:
                fancy_ui.subheader("🗺️ Geographic Locations of Matches")
                fancy_ui.map(search_results.dropna(subset=['lat', 'lon']))
        else:
            fancy_ui.warning("No matches found. Try adjusting your query keywords!")
    else:
        fancy_ui.info("💡 Pro-Tip: The AI understands context. You can type abstract phrases like 'ancient institutions' or 'schools in North America'!")