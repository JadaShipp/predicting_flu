# Predicting Flu Vaccinations

# Project purpose
The goal of this project is to use features captured from survey data to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines.

Specifically, I will be predicting two probabilities:
- h1n1_vaccine - Whether respondent received H1N1 flu vaccine.
- seasonal_vaccine - Whether respondent received seasonal flu vaccine.
> Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and not multiclass) problem.

# Executive Summary

# Plan
### 2. Prepare
  * Chekck data types
  * Get an idea of the null values
  * Impute data where possible
  * Drop columns that are not useful
  * Encode appropriately
  * Scale if needed
### 3. Explore
  * Hypothesis testing
  * Vizualize data
  * Compile list of most useful features for modeling
### 4. Modeling
  * Classifications models used:
  > - K Nearest Neighbors
  > - Random Forest
  > - Logistic Regression

Try each model on the training set, then evaluate on validation set. Tune hyperparameters as needed and finally apply only the best performing model on the test set.

- __All models will be tuned for overall accuracy as per project specifications__
### 5. Conclusions

## Acquire
The data was separated into training features and training labels. Both are needed to create and test models. Both sets can be [downloaded here](https://www.drivendata.org/competitions/66/flu-shot-learning/data/).
After dowloading them in .csv format, simply read them in with pandas and then merge them together.

## Prepare


