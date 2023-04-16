# Heart Diesease Prediction

## Introduction
This project aims to predict the risk of heart disease in individuals using their biological parameters and relevant information. The dataset used for this project contains information about 3400 individuals and their medical history. The goal of this project is to identify potential outcomes related to the risk of heart disease in the next ten years.

## Dataset

The dataset used for this project contains the following variables:

- Sex - Gender of the person
- Age - age of the person
- is_smoking - whether the person is smoking or not
- Cigs_Per_Day - Cigarettes smoked per day
- BP_Meds - Whether taking any medicine for BP or not
- Prevalent Stroke - if the patient has a history of stroke
- Prevalent Hyp - if the patient has a history of hypertension
- Diabetes - Patient is diabetic or not
- TotChol - Cholesterol Measure
- sysBP - Systolic BP
- diaBP - Diastolic BP
- BMI - Body Mass Index
- HeartRate - HeartRate Measure

## Data Preprocessing

The data was preprocessed by handling missing values and replacing them with median or mode as per the neccessities, introducing dummy variables, and using SMOTE to resample the data since the data was highly imbalanced.

## Models

Three models were used to predict the risk of heart disease: logistic regression, random forest, and KNN. After comparing the accuracy of these models, it was found that the random forest model was the best in terms of its ability to accurately predict the risk of heart disease.

## Deployment

The prediction pipeline was deployed using Flask. The application.py file contains the Flask application that accepts input from the user and returns the predicted outcome.

## File Structure

- .ebextensions: contains deployment configuration files for AWS server
- artifacts: contains the pickle file, train data and test data
- notebook: contains the Exploratory Data Analysis and the whole project to test various methods 
- src: This directory contains the source code for the project, including the pipeline and different components.
  - components: This directory contains the code for the individual components of the project, such as data ingestion, transformation, and modeling.
      - data_ingestion.py: This file contains functions for loading data from the given CSV file.
      - data_transformation.py: This file contains functions for preprocessing and transforming the data for use in the machine learning models. It includes functions for handling missing values, encoding categorical variables, and scaling the data.
      - model_trainer.py: This file contains the code to train our model and tune our hyperparameters as needed 
  - pipeline: This directory contains the implementation of the prediction pipeline, which involves loading the trained model and processing the input data for prediction.
      - train_pipeline.py: This file contains code for training the machine learning models.
      - predict_pipeline.py: This file contains code for making predictions using the trained models. 
  - utils.py: This file contains utility functions for loading and transforming the input data.
  - logger.py: This file contains the code to log each time we run the code and if a exception comes it will tell us till where the code was executed.
  - exception.py: This file raises a custom exception whenever we  encounter a error 
  - init.py: This file makes Python treat directories containing it as modules
- templates: contains HTML templates for the Flask application
- .gitignore: specifies which files to ignore when committing to Git
- README.md: provides information about the project
- application.py: contains the deployment configuration
- python: contains the data transformation code
- requirements.txt: contains the Python packages required for the project
- setup.py: installs the Python packages required for the project


## Conclusion

This project provides an intuitive solution to predicting the risk of heart disease in individuals. The random forest model was the best model for predicting the risk of heart disease. This prediction pipeline can help medical institutions determine which patients to test and whom to not given they had a huge volume of patients coming to examine.
