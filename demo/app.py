"""
Patient Analyzer - Streamlit Demo Application

This demo visualizes patient embeddings and finds nearest neighbors
using synthetic data and TF-IDF vectorization (lightweight for CI).

For production use with better embedding quality, see comments on
integrating sentence-transformers or Bio_ClinicalBERT.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt


def load_patient_data():
    """Load synthetic patient data from CSV."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sample_patients.csv")
    df = pd.read_csv(csv_path)
    return df


def create_patient_text(row):
    """
    Combine patient features into a single text for embedding.
    
    This creates a text representation that captures the key patient attributes
    for similarity comparison.
    """
    return (
        f"Patient age {row['age']} {row['gender']} "
        f"diagnosis {row['diagnosis_code']} "
        f"{row['text_note']}"
    )


def compute_tfidf_embeddings(patient_texts):
    """
    Compute TF-IDF embeddings for patient texts.
    
    TF-IDF is lightweight and fast - suitable for demos and CI.
    
    # -----------------------------------------------------------------
    # PRODUCTION ALTERNATIVE: sentence-transformers
    # For better semantic similarity, replace this function with:
    #
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(patient_texts)
    # return embeddings
    #
    # Or for medical domain, use Bio_ClinicalBERT:
    # model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
    # -----------------------------------------------------------------
    """
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    embeddings = vectorizer.fit_transform(patient_texts).toarray()
    return embeddings, vectorizer


def find_nearest_neighbors(embeddings, patient_idx, top_k=3):
    """
    Find the top-k nearest neighbors for a given patient.
    
    Uses cosine similarity which works well for text embeddings.
    """
    # Compute cosine similarity between selected patient and all others
    query_embedding = embeddings[patient_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices sorted by similarity (descending), excluding the patient itself
    similar_indices = np.argsort(similarities)[::-1]
    
    # Filter out the patient itself and get top-k
    neighbors = [(idx, similarities[idx]) 
                 for idx in similar_indices 
                 if idx != patient_idx][:top_k]
    
    return neighbors


def reduce_to_2d(embeddings):
    """
    Reduce embeddings to 2D using PCA for visualization.
    
    PCA is fast and deterministic - good for demos.
    For more nuanced visualization, consider t-SNE or UMAP.
    """
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    return embeddings_2d


def plot_patient_embeddings(df, embeddings_2d, selected_idx, neighbor_indices):
    """
    Create a 2D scatter plot of patient embeddings.
    
    Highlights the selected patient and their nearest neighbors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all patients
    ax.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c='lightgray', 
        s=100, 
        alpha=0.6,
        label='Other Patients'
    )
    
    # Highlight neighbors
    neighbor_idx_list = [idx for idx, _ in neighbor_indices]
    if neighbor_idx_list:
        ax.scatter(
            embeddings_2d[neighbor_idx_list, 0],
            embeddings_2d[neighbor_idx_list, 1],
            c='orange',
            s=150,
            alpha=0.9,
            label='Nearest Neighbors',
            edgecolors='black'
        )
    
    # Highlight selected patient
    ax.scatter(
        embeddings_2d[selected_idx, 0],
        embeddings_2d[selected_idx, 1],
        c='red',
        s=200,
        alpha=1.0,
        label='Selected Patient',
        marker='*',
        edgecolors='black'
    )
    
    # Add patient ID labels
    for i, row in df.iterrows():
        ax.annotate(
            row['patient_id'],
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_title('Patient Embeddings (2D PCA Projection)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Patient Analyzer Demo",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Patient Analyzer Demo")
    st.markdown("""
    This demo shows how to compare patients in **embedding space** and find 
    similar patients based on their medical records.
    
    ⚠️ **Note**: This uses **synthetic data only** - no real patient information.
    """)
    
    # Load data
    df = load_patient_data()
    
    # Create text representations
    patient_texts = [create_patient_text(row) for _, row in df.iterrows()]
    
    # Compute embeddings
    embeddings, _ = compute_tfidf_embeddings(patient_texts)
    
    # Reduce to 2D for visualization
    embeddings_2d = reduce_to_2d(embeddings)
    
    # Sidebar for patient selection
    st.sidebar.header("Select a Patient")
    patient_options = {f"{row['patient_id']} ({row['age']}y, {row['gender']})": i 
                       for i, row in df.iterrows()}
    selected_label = st.sidebar.selectbox(
        "Choose a patient to analyze:",
        options=list(patient_options.keys())
    )
    selected_idx = patient_options[selected_label]
    
    # Find nearest neighbors
    top_k = st.sidebar.slider("Number of similar patients:", 1, 5, 3)
    neighbors = find_nearest_neighbors(embeddings, selected_idx, top_k=top_k)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Selected Patient Details")
        selected_patient = df.iloc[selected_idx]
        st.markdown(f"""
        - **Patient ID**: {selected_patient['patient_id']}
        - **Age**: {selected_patient['age']}
        - **Gender**: {selected_patient['gender']}
        - **Diagnosis Code**: {selected_patient['diagnosis_code']}
        """)
        st.markdown("**Clinical Note:**")
        st.info(selected_patient['text_note'])
    
    with col2:
        st.subheader(f"🔍 Top-{top_k} Most Similar Patients")
        for idx, similarity in neighbors:
            neighbor = df.iloc[idx]
            st.markdown(f"""
            **{neighbor['patient_id']}** (Similarity: {similarity:.3f})
            - Age: {neighbor['age']}, Gender: {neighbor['gender']}
            - Diagnosis: {neighbor['diagnosis_code']}
            """)
            with st.expander(f"View clinical note for {neighbor['patient_id']}"):
                st.write(neighbor['text_note'])
    
    # Visualization
    st.subheader("📊 Patient Embedding Visualization (2D PCA)")
    fig = plot_patient_embeddings(df, embeddings_2d, selected_idx, neighbors)
    st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Technical Details**:
    - Embeddings: TF-IDF vectorization (lightweight for demo)
    - Similarity: Cosine similarity
    - Visualization: PCA dimensionality reduction
    
    For production, consider using `sentence-transformers` or `Bio_ClinicalBERT` 
    for better semantic similarity (see code comments).
    """)


# Export functions for testing
__all__ = [
    'load_patient_data',
    'create_patient_text',
    'compute_tfidf_embeddings',
    'find_nearest_neighbors',
    'reduce_to_2d'
]


if __name__ == "__main__":
    main()
