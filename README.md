# Patient Analyzer

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

This project demonstrates how to compare medical patients based on their medical histories using a pre-trained medical BERT model. The goal is to generate embeddings for patients' medical records, reduce their dimensionality using t-SNE, and visualize the relationships among patients.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

...

## Installation {#installation}

Create a new virtual environment and install the dependencies:

```bash
conda create -n patient-analyzer python=3.10
conda activate patient-analyzer
pip install -r requirements.txt
``` 

## Project Structure {#project-structure}
```
patient-analyzer/
│
├── main.py                  # Main script to run the project
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── data/                    # (Optional) Folder for storing any additional datasets or outputs
```


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.