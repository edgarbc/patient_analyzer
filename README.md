# Patient Analyzer

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![CI](https://github.com/edgarbc/patient_analyzer/actions/workflows/ci.yml/badge.svg)

## TL;DR

A portfolio project demonstrating how to **compare medical patients in embedding space** using machine learning. Includes an interactive Streamlit demo that visualizes patient embeddings and finds nearest neighbors using synthetic data.

## Problem Statement

In healthcare analytics, comparing patients based on their medical histories can help with:
- Finding similar patient cases for treatment insights
- Identifying patient cohorts for clinical studies
- Supporting clinical decision-making with data-driven similarity analysis

This project converts patient records (age, gender, diagnosis codes, clinical notes) into vector embeddings and uses distance metrics to find similar patients. The embeddings can be visualized in 2D using PCA/t-SNE to understand patient clusters.

## Quick Start

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/edgarbc/patient_analyzer.git
cd patient_analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit Demo

```bash
streamlit run demo/app.py
```

The demo will open in your browser and allow you to:
1. Select a patient from the synthetic dataset
2. View patient details
3. See the top-3 most similar patients
4. Visualize all patients in 2D embedding space

## Features

- **Synthetic Patient Data**: 10 sample patients with realistic attributes (age, gender, diagnosis codes, clinical notes)
- **Lightweight Embeddings**: Uses TF-IDF vectorization for fast, dependency-light processing
- **Nearest Neighbor Search**: Finds similar patients using cosine similarity
- **Interactive Visualization**: 2D PCA scatter plot showing patient clusters
- **CI-Friendly**: No large model downloads required for demo/tests

## Project Structure

```
patient_analyzer/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── requirements.txt             # Python dependencies
├── demo/
│   ├── app.py                   # Streamlit demo application
│   └── sample_patients.csv      # Synthetic patient dataset
├── tests/
│   └── smoke_test.py            # CI smoke test
├── .github/
│   └── workflows/
│       └── ci.yml               # CI workflow
├── generate_embeddings.py       # BERT-based embedding generation (production)
├── generate_patients.py         # Patient data generation utilities
├── visualize_patients.py        # Visualization utilities
└── notebooks/                   # Jupyter notebooks for exploration
```

## Using Real Embeddings (Production)

For production use cases with better similarity quality, you can use sentence transformers or medical BERT models:

```bash
# Install optional dependencies for production embeddings
pip install sentence-transformers torch transformers
```

Then modify `demo/app.py` to use the `sentence-transformers` or `Bio_ClinicalBERT` model instead of TF-IDF (see comments in the code).

## Privacy & Data Use Statement

⚠️ **Important**: This project uses **100% synthetic data** for demonstration purposes.

- **No real patient data** is included or required
- The sample dataset is algorithmically generated with fictional patient information
- For production use with real data, ensure compliance with:
  - HIPAA (US healthcare data)
  - GDPR (EU personal data)
  - Your organization's data governance policies

**Never commit real patient data to version control.**

## Expected Output

The Streamlit demo shows a patient selector, patient details table, nearest neighbors list, and a 2D scatter plot of patient embeddings.

![Demo Screenshot](https://github.com/user-attachments/assets/358ac05e-0fb8-453c-b51a-3e22f7ed269a)

## Development

### Running Tests

```bash
python -m pytest tests/smoke_test.py -v
```

### Linting

```bash
# Install development dependencies
pip install flake8 black

# Run linter
flake8 demo/ tests/
black --check demo/ tests/
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.