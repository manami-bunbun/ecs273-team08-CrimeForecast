# Implementation Submission Description
https://docs.google.com/document/d/1_WtcIEK9NZbwHpl8Ubc147YtbOsQDxEGr36g5S8PsCQ/edit?tab=t.0#heading=h.rgiyidgnbo62

# Implementation Plan

This project is structured into two main components:

1. `ml_evaluation/` — Responsible for model training, LLM-enhanced feature engineering, and performance evaluation.

2. `web_app/` — A fullstack web application for interactive visualization and explanation using model predictions and LLM output.

---


## 📁 data/

```
data/
├── raw/                        # Original CSV files (e.g., SF crime data)
├── processed/                  # Cleaned & feature-engineered datasets for ML
└── external/                   # Event/context data for LLM input (e.g., news headlines)
```

---

## 📁 ml_evaluation/

```
ml_evaluation/
├── notebooks/                  # Jupyter Notebooks for EDA and prototyping
│   ├── 01_explore.ipynb        # Explore data distribution, spatial & temporal patterns
│   ├── 02_rf_model.ipynb       # Build and tune Random Forest baseline
│   └── 03_llm_feature_gen.ipynb# Prototype LLM-based feature generation
│
├── utils/                      # Utility functions for preprocessing, metrics, configs
│   ├── preprocessing.py        # Data cleaning, time formatting, missing value handling
│   ├── config.py               # Global paths, constants, model parameters
│   └── metrics.py              # Custom evaluation metrics (e.g., F1 score, confusion matrix)
│
├── baseline_rf.py              # Random Forest training on structured features
├── llm_features.py             # LLM-assisted feature engineering module (e.g., event tags, imputation)
├── evaluate_models.py          # Run both models, compare accuracy, generate results
├── results/                    # Output folder for model predictions and evaluation visualizations
```

---

## 📁 web_app/

```
web_app/
├── backend/                    # FastAPI server that connects model output and LLM explanations
│   ├── app.py                  # Entry point for running the FastAPI app
│   ├── routes/
│   │   ├── prediction.py       # API endpoint to serve model predictions
│   │   └── explanation.py      # API endpoint to serve LLM-generated explanations
│   └── utils/                  # Helper functions for loading models and formatting responses
│
├── frontend/                   # Frontend interface using HTML, JS (e.g., D3.js or React)
│   ├── index.html              # Web page structure
│   ├── js/
│   │   ├── map.js              # Map-based crime visualization
│   │   └── panel.js            # Narrative panel for LLM-generated summaries
│   └── style.css               # Styling for layout, components, and map
```


---

## 📄 Supporting Files

```
requirements.txt               # Python dependencies for ML, API, and LLM
README.md                      # Project overview, setup guide, usage instructions
Implementation_Plan.md         # This document: file structure and purpose overview
.env (copy .env.example)
```
