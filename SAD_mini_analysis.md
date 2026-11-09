# SAD Mini-Analysis — Wine Quality Prediction System

## 1. Problem statement
A wine producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements. Manual review may be time-consuming, so the prototype provides an initial classification to support further quality assessment.

## 2. Intended users
- Quality-control staff
- Production supervisor
- Data analyst

## 3. Inputs
The system accepts these 11 numerical predictor attributes:
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

The historical `quality` score is used only to derive the training target. It must not be included as an input feature.

## 4. Process
1. Retrieve and inspect the UCI red-wine dataset.
2. Remove duplicate records and derive a binary target: score 6 or higher is GOOD QUALITY; below 6 is NEEDS IMPROVEMENT.
3. Split the data into stratified training and testing sets.
4. Impute missing numeric values using the median and scale the features.
5. Train a Random Forest classifier.
6. Accept an uploaded CSV, validate required columns, and apply the trained model.
7. Display and allow download of prediction results.

## 5. Outputs
- Predicted label: `GOOD QUALITY` or `NEEDS IMPROVEMENT`
- Model evaluation metrics: accuracy, precision, recall, and F1-score
- Downloadable CSV containing the original uploaded fields and prediction

## 6. Key limitation
The model is trained on a particular public red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. It is a classroom prototype and its predictions are not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
