# Multimodal Data Preprocessing Project

## Project Overview

This repository contains a complete multimodal machine learning workflow for a user-authentication and product-recommendation system. The project combines:

- tabular customer data,
- facial image data,
- and audio voice samples,

to build three different models:

1. a facial recognition model,
2. a voiceprint verification model,
3. and a product recommendation model.

The system simulates a secure decision flow where a user must pass face and voice authentication before a product recommendation is produced.

---

## What This Repository Contains

The project now includes a full end-to-end structure:

- a Jupyter notebook for preprocessing and experimentation,
- a Python command-line application for running the full authentication and prediction flow,
- trained model files,
- engineered datasets,
- utility modules for face and voice processing,
- and a dependency list for installation.

---

## Repository Structure

```text
MLPipeline-Formative2-DataPreprocessing/
├── app.py
├── requirements.txt
├── multimodal_data_preprocessing.ipynb
├── README.md
├── feature_engineered datasets/
│   ├── audio_features.csv
│   ├── image_features.csv
│   └── merged_customer_data_with_engineerd_features.csv
├── models/
│   ├── facial_recognition_rf_model.pkl
│   ├── product_recommendation_model.pkl
│   └── voiceprint_verification_model.pkl
├── test_samples/
└── utils/
    ├── face_utils.py
    ├── prediction.py
    └── voice_utils.py
```

---

## Assignment Goal

The main objectives of the assignment are to:

1. merge social profile and transaction data,
2. clean and engineer features for machine learning,
3. collect and preprocess image and audio data,
4. build face and voice authentication models,
5. train a product recommendation system,
6. and demonstrate a multimodal approval workflow.

---

## Dataset Sources

The notebook uses two main sources of tabular data:

- customer social profiles
- customer transactions

These datasets are merged using a join key based on customer identifiers after cleaning and standardizing the IDs. The merged dataset is then used to train the product recommendation model.

## Workflow Summary

### 1. Data Merge and Feature Engineering

The first stage of the project focuses on preparing a unified customer dataset.

Key steps included:

- loading the social and transaction datasets,
- inspecting shape, columns, and data types,
- checking for missing values and duplicate rows,
- standardizing customer identifier formats,
- performing an inner join between the two datasets,
- cleaning the merged dataset, and
- engineering new features for downstream modeling.

The notebook reports that the initial merge produced 219 records and that duplicate removal reduced the final merged dataset to 213 rows.

Feature engineering included:

- purchase interest score,
- review sentiment indicators,
- transaction timing features such as month, day, weekday, and weekend flag,
- and other behavioral and transactional attributes used by the product model.

The cleaned merged dataset was saved as a CSV for reuse and further analysis.

### 2. Image-Based Authentication

The image processing stage focuses on facial recognition using embeddings extracted from face images.

The notebook includes:

- collection of facial images for multiple individuals,
- display of sample images,
- detection and extraction of facial regions,
- generation of FaceNet embeddings,
- and training of a facial recognition classifier.

The pipeline uses a Random Forest classifier on the extracted embeddings to classify the identity of a person from an image.

The face model achieved excellent performance on the available dataset, with:

- accuracy: 100%
- precision: 100%
- recall: 100%
- F1-score: 100%

This indicates that the extracted facial embeddings were highly discriminative for the small dataset used in the notebook.

### 3. Audio-Based Authentication

The audio processing stage uses speech recordings to build a voiceprint verification system.

The notebook includes:

- loading and displaying audio samples as waveforms and spectrograms,
- applying augmentations such as pitch shift, time stretch, and added background noise,
- extracting handcrafted acoustic features such as MFCCs, spectral centroid, spectral bandwidth, spectral roll-off, zero-crossing rate, and RMS energy,
- and saving the resulting feature matrix to a CSV file.

These audio features were then used to train a Random Forest voice classifier for speaker verification.

The voice verification model achieved strong results, with a mean cross-validation accuracy of 100% across folds, showing that the extracted audio descriptors were highly effective for the notebook’s sample size.

### 4. Product Recommendation Model

The tabular merged dataset was used to build a product recommendation classifier that predicts the likely product category for a customer.

Several algorithms were evaluated, including:

- Random Forest,
- XGBoost,
- Logistic Regression, and
- LightGBM.

The best-performing model was Random Forest, which achieved:

- accuracy: 81.40%
- precision: 83.40%
- recall: 81.40%
- F1-score: 81.48%

These results indicate that the engineered merged dataset contained strong predictive signals for classifying product categories.

### 5. Multimodal Authentication Demo

The final application, app.py, runs a complete interactive workflow:

- Step 1: face recognition,
- Step 2: voice verification,
- Step 3: product recommendation.

The system only proceeds if both biometric checks pass and the face and voice belong to the same identity.

---

## How to Run the Project

Follow these steps in order.

### Step 1: Clone the repository

```bash
git clone <repo-url>
cd MLPipeline-Formative2-DataPreprocessing
```

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install the dependencies

```bash
pip install -r requirements.txt
```

> The requirements file includes the libraries needed for data processing, machine learning, image analysis, audio processing, and model loading.

### Step 4: Open the notebook (optional)

You can inspect the full workflow in:

```text
multimodal_data_preprocessing.ipynb
```

Run the notebook cells in order if you want to reproduce the preprocessing, feature extraction, and model training steps.

### Step 5: Run the command-line application

Once the environment is ready, run:

```bash
python app.py
```

The program will prompt you for:

- a face image path,
- a voice recording path in WAV format,
- and then the product prediction inputs.

### Step 6: Use sample files if needed

The repository contains a test_samples folder for trying the application. If you have your own files, provide their full paths when prompted.

---

## How the CLI Application Works

When you run app.py, the program will:

1. load the trained face, voice, and product models,
2. ask for a face image,
3. extract face embeddings and verify identity,
4. ask for a voice recording,
5. extract voice features and verify the speaker,
6. ensure the face and voice belong to the same user,
7. and finally request transaction details to produce a product prediction.

If face recognition or voice verification fails, the application stops and shows an access denied message.

---

## Important Notes

- The voice input must be a WAV file.
- The face input should be an image file that contains a visible face.
- The trained models are already stored in the models directory, so you can run the app without retraining immediately.
- If you want to retrain models, use the notebook as the main reference.

---

## Expected Output

A successful run will:

- recognize the face,
- verify the voice,
- confirm identity,
- and print a predicted product category.

---

## Summary

This repository provides a full multimodal pipeline for:

- data preprocessing,
- image and audio feature extraction,
- biometric-style authentication,
- and product recommendation.

It includes both the notebook-based development workflow and the runnable command-line application for demonstration.


## TEAM NAMES 

1.Francis MUTABAZI 

2.Gaddie Irakoze

3.Gabriel TUYISINGIZE SEZIBERA
   
4.Carla BATONI
