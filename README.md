# Spaceship Titanic - Cosmic Passenger Transportation Classification

This repository contains an end-to-end Machine Learning classification pipeline developed for Kaggle's **"Spaceship Titanic"** competition. The objective is to predict whether a passenger was transported to an alternate dimension following an interstellar collision, utilizing records recovered from the ship's damaged computer network.

## Project Overview
This project tackles a high-dimensional binary classification problem utilizing complex tabular parsing, group dynamics extraction, and a meta-learning stacking ensemble framework. The pipeline achieves robust generalizability by validating via Stratified K-Fold Cross-Validation.

## Engineering & Pipeline Architecture

### 1. Granular Feature Engineering
* **Passenger Group Dynamics**: Parsed `PassengerId` (`gggg_pp`) to extract unique travel `Group` IDs, calculating exact `GroupSize` and generating a binary `IsSolo` indicator to capture socio-travel behavior.
* **Structured String Parsing**: Deconstructed the `Cabin` feature (`deck/num/side`) into independent variables (`Cabin_deck`, `Cabin_num`, `Cabin_side`) to unlock spatial risk signals.
* **Expenditure Consolidation**: Aggregated billing values across luxury amenities (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) into a total log-scaled `TotalSpending` feature alongside an engineered `IsSpender` binary flag.

### 2. Contextual Imputation & Preprocessing
* Imputed `Age` values utilizing median values segmented conditionally by `HomePlanet`.
* Replaced structural categorical missingness via local mode tracking, followed by continuous handling of skewed distributions using logarithmic transformations.

### 3. Stacking Ensemble Architecture
To maximize classification boundaries, the pipeline uses a layered **Stacking Classifier** framework:
* **Base Learners**: Random Forest Classifier, XGBoost Regressor, and LightGBM Classifier.
* **Meta-Learner**: Logistic Regression (regularized to optimize out-of-fold base learner classification probabilities).
* **Validation**: 5-Fold Stratified Cross-Validation ensuring rigid class distribution boundaries.

## Workflow Summary
| Step | Process Type | Target/Features |
|---|---|---|
| 1 | Data Loading | Merging train/test shapes cleanly |
| 2 | String Deconstruction | ID Group sizes, Cabin Deck & Side extraction |
| 3 | Preprocessing | Log-scaling expenditures, robust scaling numeric features |
| 4 | Stacking Ensemble | RF + XGB + LGBM mapped into Logistic Regression |
