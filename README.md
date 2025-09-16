![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Robust Universal EDA + Feature Engineering Assistant
A Streamlit-based tool for robust exploratory data analysis (EDA) of any CSV dataset, with AI-powered recommendations for feature engineering and modeling guidance. It also implements the features automatically to the dataset ready for download and analysis.

## Features
* Automatic preprocessing of mixed-type columns (numeric, categorical, datetime, nested objects).
* Robust EDA for numeric and categorical features:
    * Distribution plots
    * Correlation heatmaps
    * Pairplots (sampled for large datasets)
    * Target-aware analysis (numeric or categorical target)
    * Feature vs target visualizations (boxplots, scatterplots, countplots)
    * Missing value reporting with counts and percentages.

* AI Recommendations via GPT-OSS120B:
    * Suggest additional engineered features
    * Highlight problematic columns
    * Guided by user-provided scenario context
    * Handles large datasets efficiently via sampling and plot limits.
    * Implementation of suggested features automatically into a downloadable .csv

## Installation

Clone this repository:
```
git clone https://github.com/yourusername/robust-eda-groq.git
cd robust-eda-groq
```

Create a Python virtual environment:
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

Install dependencies:
```
pip install -r requirements.txt
```

Dependencies include: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, groq.

## Set your Groq API key in your environment:
```
export GROQ_API_KEY="your_api_key_here"  # Linux/macOS
set GROQ_API_KEY=your_api_key_here       # Windows
```
## Usage

Run the Streamlit app:
```
streamlit run main.py
```
* Upload your CSV dataset.
* Select the dependent (target) column.
* Optionally, provide a comma-separated list of feature columns.

Configure plot limits:
* Max features to plot
* Pairplot sample size

Click *Run EDA* to explore your dataset.

Optionally, provide a scenario description and click *Ask Feature Engine for Recommendations* for AI-assisted guidance.

## Features of the EDA

* Numeric Columns:
    * Descriptive statistics (mean, std, min, max, etc.)
    * Distribution plots and KDE
    * Correlation heatmaps
    * Sampled pairplots

* Categorical Columns:
    * Unique values, mode
    * Barplots of top 50 categories

* Target-Centric Analysis:
    * Target distribution
    * Numeric features vs target (scatter or boxplots)
    * Categorical features vs target (countplots or boxplots)

* Missing Values Analysis:
    * Count and percentage of missing data per column

## Feature Engine Recommendations

* Uses the Groq API to suggest:
   * Additional engineered features
   * Problematic columns
* Automatically implements engineered features into the dataset and transforms the data for download.

User provides:
* Dependent column
* Feature list
* Scenario/context for AI guidance

## Project Structure
```
eda-feature-engineer-automation/
├── main.py                  # Streamlit main app
├── requirements.txt        # Python dependencies
├── README.md               # GitHub documentation
└── utils/
    └── preprocessing.py    # Add all preprocessing and helper functions
```

## Notes
* Large datasets: Sampling and plot limits prevent slow rendering.
* All object-like or nested columns are converted to strings/JSON for compatibility.
* Target column is required for robust, target-aware EDA.

## Contributing
* Fork the repository
* Create a new branch (git checkout -b feature-name)
* Make your changes
* Submit a pull request

Please ensure code is well-documented and functions remain modular

License
This project is licensed under the MIT License.
