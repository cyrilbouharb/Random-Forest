# README for COMPSCI 589 Homework 3

## Overview
This homework focuses on implementing and analyzing the Random Forest algorithm across two datasets: the Wine Dataset and the 1984 United States Congressional Voting Dataset. The analysis explores how the number of trees in the forest (ntree) affects the algorithm's performance metrics: accuracy, precision, recall, and F1 score. In addition to the main tasks, this homework includes three extra credit assignments that further extend the analysis. The code is developed in Google Colab for an interactive analysis experience and is also available in Python script format.

## Environment Setup
Before running the scripts, please ensure you have a Python 3 environment set up with the necessary libraries installed. 

## Note for TAs
The original code was developed in Google Colab (notebook), it has been converted into Python scripts for submission as requested. To review the code's logic and the execution flow please see the notebook (.ipynb) version because this is how I was working on it, and it's easier to understand. I answer the questions in the notebook and write the code too.

## Datasets Analyzed
Wine Dataset: Aims to classify wine types based on chemical contents. It includes 178 instances with 13 numerical attributes and 3 classes. </br>
1984 United States Congressional Voting Dataset: Aims to predict the party affiliation of U.S. House of Representatives members based on their voting behavior. It includes 435 instances with 16 categorical attributes and 2 classes.

## Tasks
Random Forest Analysis: For each dataset, train random forests with ntree values of 1, 5, 10, 20, 30, 40, and 50. Evaluate each forest using stratified cross-validation. </br>
Performance Metrics: Measure accuracy, precision, recall, and F1 score for each ntree value. </br>
Graphical Analysis: Create graphs showing how each performance metric varies with ntree for both datasets, resulting in a total of 8 graphs.</br>
ntree Selection Discussion: Discuss the optimal ntree value for real-life deployment based on the analysis. </br>
Performance Impact Discussion: Analyze how changing ntree values affects each metric and the overall performance of the random forest.

## Extra Credit Tasks
Gini Criterion Analysis: Replicate the analysis using the Gini criterion for node splitting. </br>
Breast Cancer Dataset Analysis: Apply the random forest algorithm to classify biopsy samples for breast cancer detection. </br>
Contraceptive Method Choice Dataset Analysis: Analyze a more challenging dataset combining numerical and categorical attributes to predict contraceptive method choice.

### Part 1: Random Forest on Wine Dataset
**File Name: 'hmw3_wine.py' or the notebook version for interactive visualization 'hmw3_wine.ipynb'.**
This script applies the Random Forest algorithm to the Wine Dataset to classify wine types based on chemical properties. It evaluates the model across a range of ntree values and generates performance metrics. <br />
To run this script, use the following command in your terminal: **python hmw3_wine.py**

### Part 2: Random Forest on Congressional Voting Dataset
**File Name: 'hmw3_congressionalvoting.py' or the notebook version for interactive visualization 'hmw3_congressionalvoting.ipynb'.**
Implements Random Forest to predict party affiliation in the 1984 United States Congressional Voting Dataset. Analysis includes performance evaluation for varied ntree settings. <br />
To run this script, use the following command in your terminal: **python hmw3_congressionalvoting.py**

### Extra Credit 1: Gini Criterion on Congressional Voting and Wine Datasets
**File Name: 'hmw3_extracredit1.py' or the notebook version for interactive visualization 'hmw3_extracredit1.ipynb'.**
Adapts the Random Forest algorithm to use the Gini criterion for splitting nodes, assessed on the Congressional Voting Dataset and the Wine Dataset. <br />
To run this script, use the following command in your terminal: **python hmw3_extracredit1.py**

### Extra Credit 2: Random Forest on Breast Cancer Dataset
**File Name: 'hmw3_extracredit2.py' or the notebook version for interactive visualization 'hmw3_extracredit2.ipynb'.**
This script applies Random Forest to classify biopsy samples in the Breast Cancer Dataset, with an in-depth analysis of ntree's impact. <br />
To run this script, use the following command in your terminal: **python hmw3_extracredit2.py**

### Extra Credit 3: Random Forest on Contraceptive Method Choice Dataset
**File Name: 'hmw3_extracredit3.py' or the notebook version for interactive visualization 'hmw3_extracredit3.ipynb'.**
This script focuses on predicting contraceptive method choice using Random Forest, combining numerical and categorical attributes for a complex analysis.<br />
To run this script, use the following command in your terminal: **python hmw3_extracredit3.py**