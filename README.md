# Multimodal Data Preprocessing

## Assignment Overview

This repository documents a multimodal machine learning pipeline for a user-authentication and product-recommendation system. The project combines tabular customer data, facial images, and audio samples to build three complementary models:

- a product recommendation model,
- a facial recognition model, and
- a voiceprint verification model.

The overall workflow simulates a real-world decision process in which a user must pass face and voice verification before a transaction or product suggestion is approved.

## Project Goal

The objective of this assignment is to:

1. merge customer social and transaction data into a single predictive dataset,
2. preprocess and engineer features for machine learning,
3. collect and process image and audio data,
4. build authentication models for face and voice verification,
5. train and evaluate a product recommendation model, and
6. demonstrate a multimodal decision flow for authorized and unauthorized access attempts.

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

### 5. Multimodal Decision Logic

The notebook combines the three trained systems into a multimodal authentication and recommendation workflow:

- face recognition checks whether the user matches a known registered identity,
- voiceprint verification checks whether the provided voice sample belongs to the same user,
- and the product model decides whether the requested recommendation or transaction should be approved.

This creates a security-style decision flow in which a user must pass the biometric checks before proceeding to the recommendation step.

## Repository Contents

- [multimodal_data_preprocessing.ipynb](multimodal_data_preprocessing.ipynb) — the full notebook containing the preprocessing, modeling, and evaluation workflow.
- [model](model) — trained model artifacts for the facial recognition, voiceprint verification, and product recommendation pipelines.
- [README.md](README.md) — project documentation.

## Files Produced During the Workflow

The notebook generates several outputs, including:

- merged customer data CSVs,
- image feature extraction outputs,
- audio feature extraction outputs,
- and trained model files stored in [model](model).

## How to Use This Repository

1. Open [multimodal_data_preprocessing.ipynb](multimodal_data_preprocessing.ipynb) in Jupyter or VS Code.
2. Run the notebook cells in order to reproduce the preprocessing, model training, and evaluation steps.
3. Review the trained models stored in [model](model) for inference or further experimentation.
4. Use the notebook’s workflow as a foundation for extending the system into a command-line demo or a web-based application.

## Notes

This project is a prototype implementation aimed at demonstrating how multimodal data preprocessing and authentication can be combined for a realistic decision-support system. The notebook provides the core pipeline and model evaluation logic, while the trained model files make the workflow easier to reuse.

## Summary

This assignment was completed by building a complete preprocessing and modeling pipeline that:

- merges tabular customer datasets,
- prepares image and audio data,
- creates biometric-style authentication models,
- trains a product recommendation model,
- and documents the end-to-end multimodal workflow in a reproducible notebook.
