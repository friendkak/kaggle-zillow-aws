### Zillow Kaggle Challenge Modification

This project is designed to help Zillow by predicting the log difference between ground truth sale price and Zillow's sale price estimate.

## Real estate housing price prediction
Using statistical methods to predict real estate housing prices goes back many years.  One of the seminal machine learning datasets is the Boston Housing Price Dataset.  Over the course of a number of years, it has been found that Generalized Linear Models give the best performance on this task, particularly with gamma or inverse gaussian link functions (https://irudnyts.github.io/dortmund-real-estate-market-analysis-glm-and-gam/).  To my knowledge, most real-estate industry pricing models are GLMs, and it's likely that Zillow's is also, or at least largely derived from it.

## Backfitting
The specific task is to re-predict the loss of Zillow's models.  Zillow did not disclose the actual sale price in the data it released, but only the error of its model.  Thus, the modeling task here is based both on the quirks of the housing market and the quirks of Zillow's model.  The process of fitting a model to the errors of another model is called 'backfitting.'

## Data source
This task was a Kaggle competition in 2017 (?}, so datasets (including a holdout test set) are pre-defined.  In addition, Kaggle provides a leaderboard with scoring history that allows us to see how our solutions compare to the leading solutions at the time of the competition and thereafter.

## Task and approach
The data are already curated by Kaggle.  We will first do a study of data integrity and decide if any columns are substantially missing.  This is a regression task, so we will begin by using the Sagemaker built-in models.  Then we will try using GLM models using either R or H2O.  Finally, we will try a feed-forward NN using torch or tensorflow.  The team will split of work based on modelling technique.  While the goal of the project is to find a competitive model for the Kaggle leaderboard, part of the study is to attempt to see whether a deep learning approach is superior to a feature engineering approach.
