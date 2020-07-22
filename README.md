# Predicting Flu Vaccinations

# Project purpose
The goal of this project is to use features captured from survey data to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines.

Specifically, I will be predicting two probabilities:
- h1n1_vaccine - Whether respondent received H1N1 flu vaccine.
- seasonal_vaccine - Whether respondent received seasonal flu vaccine.
> Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and not multiclass) problem.

# Plan
1. Aquisition
  * Download data into local drive
2. Prepare
  * Read in data csv using pandas
  * Chekck data types and null values
  * Fill in nulls
  * Encode appropriately
  * Scale if needed
3. Explore
4. Modeling
5. Conclusions