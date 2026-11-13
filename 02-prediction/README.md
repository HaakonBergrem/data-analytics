# Enhetsregister Prediction

Small predictive analysis of Norwegian company data from **Brønnøysundregistrene**. The models are trying to predict the industry a given company is a part of. 


## What it does

- Loads a subset of relevant columns from the raw CSV
- Cleans and prepares the data 
- Trains a baseline model, a logistic regression model, and a random forest model on relevant features from the dataset. 
- Visualises the accuracy of the models after testing, and feature importance for the random forest model (what features are best at predicting the industry of the company)


## A note on the accuracy of the models

As seen in the outputs, both models perform noticeably better than the baseline. Notably, the baseline's F1-score is much lower than its accuracy. This discrepancy occurs because the number of companies across the selected industries is imbalanced. Relying solely on accuracy would mask this class imbalance and provide a misleadingly optimistic view of the baseline's true predictive performance.

Looking at the feature importance for the random forest model, it is clear that the age of the company is the greatest predictor of the industry a company is a part of. This could be plausible since new industries emerge/are more selected for
across time. However, random forest favors continous numeric features. The only other such feature is the number of employees, and most companies in the dataset are registered with zero.  


## How to run

- Follow the instructions in the README in the main folder of the repository
- Run the "main.py" file in this folder
