# Alert-Grouping-Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20database-informational)](https://qdrant.tech/) 

[![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/Numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

## Overview

This project evaluates the performance of various embedding models for alert grouping based on Prometheus alerts in multiple languages (German, English, Spanish, and French). It aims to compare monolingual and multilingual embedding models in their ability to group similar alerts across languages.

## Key Features

- Support for multiple languages: English, German, French, and Spanish
- Evaluation of both monolingual and multilingual embedding models
- Integration with Qdrant vector database for efficient similarity search
- Comprehensive metrics calculation and evaluation
- GUI-based results visualization

## How It Works

### Alert Grouping Process

1. Alert data is loaded from CSV files for each supported language.
2. For each alert, a vector representation is created using the specified embedding model.
3. The vector is inserted into a Qdrant collection for efficient similarity search.
4. For each new alert:
   - Similar alerts are found using Qdrant's similarity search.
   - If similar alerts are found, the new alert is grouped with them.
   - If no similar alerts are found, a new group is created.
5. A final pass is made to merge any singleton groups with their most similar groups.

### Evaluation Process

1. The grouped alerts are compared against a validation set of pre-defined correct groupings.
2. Multiple clustering evaluation metrics are calculated to assess the quality of the groupings.
3. Results are displayed in a GUI, showing the performance of each model across different languages.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Adjusted Rand Index (ARI) | Measures similarity between two clusterings, adjusted for chance |
| Normalized Mutual Information (NMI) | Quantifies mutual information between cluster assignments and ground truth |
| Adjusted Mutual Information (AMI) | Similar to NMI but adjusted for chance |
| V-measure | Harmonic mean of homogeneity and completeness |
| Pairwise Precision | Proportion of pairs correctly grouped together |
| Pairwise Recall | Proportion of true pairs that were grouped together |
| F1 Score | Harmonic mean of precision and recall |

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/username/Alert-Grouping-Evaluation.git
   cd Alert-Grouping-Evaluation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Qdrant is running and accessible.

4. Configure the application in `src/config/app.py`.

5. Run the application:
   ```bash
   python app.py
   ```
## Results Interpretation

The GUI displays a comparison matrix of different models and languages, allowing for:

- Comparison of overall performance across embedding models
- Analysis of model performance for each language
- Identification of the best-performing model for multilingual alert grouping
- Assessment of model generalization across languages

Higher scores in most metrics, particularly the F1 score, indicate better grouping performance.
