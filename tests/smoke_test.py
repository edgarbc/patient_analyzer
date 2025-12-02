"""
Smoke test for Patient Analyzer demo.

This test validates that the core demo functionality works
without requiring a full Streamlit server or large model downloads.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


def test_load_patient_data():
    """Test that patient data loads correctly."""
    from demo.app import load_patient_data
    
    df = load_patient_data()
    
    # Check we have expected columns
    expected_columns = ['patient_id', 'age', 'gender', 'diagnosis_code', 'text_note']
    assert all(col in df.columns for col in expected_columns), \
        f"Missing columns. Expected: {expected_columns}, Got: {list(df.columns)}"
    
    # Check we have 10 patients
    assert len(df) == 10, f"Expected 10 patients, got {len(df)}"
    
    # Check data types
    assert df['age'].dtype in [np.int64, int], "Age should be integer"
    assert df['gender'].isin(['M', 'F']).all(), "Gender should be M or F"


def test_create_patient_text():
    """Test patient text creation."""
    from demo.app import create_patient_text
    import pandas as pd
    
    row = pd.Series({
        'age': 45,
        'gender': 'M',
        'diagnosis_code': 'E11.9',
        'text_note': 'Test note about diabetes'
    })
    
    text = create_patient_text(row)
    
    # Check that all key information is included
    assert '45' in text, "Age should be in text"
    assert 'M' in text, "Gender should be in text"
    assert 'E11.9' in text, "Diagnosis code should be in text"
    assert 'diabetes' in text, "Note content should be in text"


def test_compute_tfidf_embeddings():
    """Test TF-IDF embedding computation."""
    from demo.app import compute_tfidf_embeddings
    
    texts = [
        "Patient with diabetes and hypertension",
        "Patient with asthma and allergies",
        "Patient with heart disease"
    ]
    
    embeddings, vectorizer = compute_tfidf_embeddings(texts)
    
    # Check dimensions
    assert embeddings.shape[0] == 3, "Should have 3 embeddings"
    assert embeddings.shape[1] > 0, "Embeddings should have features"
    
    # Check that embeddings have non-zero values
    assert np.any(embeddings > 0), "Embeddings should have non-zero values"


def test_find_nearest_neighbors():
    """Test nearest neighbor search."""
    from demo.app import find_nearest_neighbors
    
    # Create simple embeddings where we know the answer
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # Patient 0
        [0.9, 0.1, 0.0],  # Patient 1 - very similar to 0
        [0.0, 1.0, 0.0],  # Patient 2 - different
        [0.0, 0.0, 1.0],  # Patient 3 - different
    ])
    
    neighbors = find_nearest_neighbors(embeddings, patient_idx=0, top_k=2)
    
    # Patient 1 should be the nearest neighbor to Patient 0
    assert len(neighbors) == 2, "Should return 2 neighbors"
    assert neighbors[0][0] == 1, "Patient 1 should be the nearest neighbor to Patient 0"
    assert neighbors[0][1] > 0.8, "Similarity should be high"


def test_reduce_to_2d():
    """Test PCA dimensionality reduction."""
    from demo.app import reduce_to_2d
    
    # Create random embeddings
    np.random.seed(42)
    embeddings = np.random.rand(10, 50)
    
    embeddings_2d = reduce_to_2d(embeddings)
    
    # Check output dimensions
    assert embeddings_2d.shape == (10, 2), f"Expected (10, 2), got {embeddings_2d.shape}"


def test_full_pipeline():
    """Integration test for the full embedding pipeline."""
    from demo.app import (
        load_patient_data,
        create_patient_text,
        compute_tfidf_embeddings,
        find_nearest_neighbors,
        reduce_to_2d
    )
    
    # Load data
    df = load_patient_data()
    
    # Create text representations
    patient_texts = [create_patient_text(row) for _, row in df.iterrows()]
    assert len(patient_texts) == 10
    
    # Compute embeddings
    embeddings, _ = compute_tfidf_embeddings(patient_texts)
    assert embeddings.shape[0] == 10
    
    # Find neighbors for first patient
    neighbors = find_nearest_neighbors(embeddings, patient_idx=0, top_k=3)
    assert len(neighbors) == 3
    
    # Reduce to 2D
    embeddings_2d = reduce_to_2d(embeddings)
    assert embeddings_2d.shape == (10, 2)
    
    print("✅ Full pipeline test passed!")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
