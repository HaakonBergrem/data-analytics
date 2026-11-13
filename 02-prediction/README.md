# Enhetsregister Prediction

Small predictive analysis of Norwegian company data from **Brønnøysundregistrene**. The models are trying to predict the industry a given company is a part of. 


## What it does

- Loads a subset of relevant columns from the raw CSV
- Cleans and prepares the data 
- Trains a baseline model, a logistic regression model, and a random forest model on relevant features from the dataset. 
- Visualises the accuracy of the models after testing, and feature importance for the random forest model (what features are best at predicting the industry of the company)

## How to run

- Follow the instructions in the README in the main folder of the repository
- Run the "main.py" file in this folder
