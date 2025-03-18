import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv')
heartDisease.replace('2', np.nan, inplace=True)

# Display data
print('Few examples from the dataset:')
print(heartDisease.head())

# Model Bayesian Network
model = BayesianModel([
    ('age', 'trestbps'),
    ('age', 'tbs'),
    ('sex', 'trestbps'),
    ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('fba', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum Likelihood Estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

# Computing the Probability of HeartDisease given Age
print('\n1. Probability of HeartDisease given Age=30')
q_HeartDisease = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q_HeartDisease)

# Computing the Probability of HeartDisease given Cholesterol
print('\n2. Probability of HeartDisease given Cholesterol=100')
q_HeartDisease = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q_HeartDisease)
