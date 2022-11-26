## Introduction

The objective of this project is to train, build, test and select the best regression model to predict the price to sell used Toyota cars, considering their year, transmission, mileage, fuel type, tax, mpg, and engine size. I am using the streamlit share for the automatic deployment of a machine-learning web app.

I wanted to ease the web app development and [Streamlit](https://www.streamlit.io/) made this possible [1]. It is an open-source library that focuses on data science and ML web app development.

## Selection of Data

The data preprocessing steps are conducted using a Jupyter Notebook and is available [here](https://github.com/vod6atwit/predict_price_used_Toyota/blob/master/preprocessing.ipynb)

All the models training, building, testing, and selecting are conducted using a Jupyter Notebook and is available [here](https://github.com/vod6atwit/predict_price_used_Toyota/blob/master/Models/regression%20models.ipynb)

The data has over 6500 samples with 8 independent/feature variables: model, year, transmission, mileage, fuelType, tax, mpg, and engineSize with 1 dependent variable: Price

The objective is to indicate the price to sell used Toyota cars.

The dataset can found online at [kaggle](https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw)[2].

### Data overview after processing:

![data overview screenshot](./img/data_overview_01.png)

### Data info after processing:

![data info screenshot](./img/data_overview_02.png)

Note that data has categorical features in 3 cols: model, transmission, and fuelType.

I used OneHotEncoder/ColumnTransformer on these features and kept the rest of the features as is. When using a Random Forest Regression model, the r2_score always over 95%.

I finally saved the model via joblib to be used for predictions by the web app.

## Methods

### Tools:

- NumPy, Pandas, and Scikit-learn for data analysis and inference
- Streamlit (st) for web app design
- GitHub and Heroku for web app deployment and hosting/version control
- VS Code as IDE

### Inference methods used with Scikit-learn:

#### - Modules: metrics, compose, preprocessing, model_selection

#### - Classes: ColumnTransformer, [OneHotEncoder](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)[3], StandardScaler, train_test_split, r2_score, max_error, mean_absolute_error, mean_squared_error

#### - Multiple Linear Regression model

- y = b0 + b1x1 + b2x2 + ... + bNxN
- [Simple_and_multiple_linear_regression](https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression)[4]
- [backward-elimination](https://www.simplilearn.com/what-is-backward-elimination-technique-in-machine-learning-article#:~:text=What%20is%20backward%20elimination%20in,is%20removed%20from%20the%20model.)[5]
- [p-value_01](https://www.investopedia.com/terms/p/p-value.asp)[6]
- [p-value_02](https://www.simplypsychology.org/p-value.html)[7]

#### Support Vector Regression (SVR) model

- [SVR](https://files.core.ac.uk/pdf/2612/81523322.pdf)[8]
- [svm-kernel-functions](https://data-flair.training/blogs/svm-kernel-functions/)[9]

#### Decision Tree Regression model

- [entropy-information-gain-machine-learning](https://www.section.io/engineering-education/entropy-information-gain-machine-learning/)[10]
- [entropy-how-decision-trees-make-decisions](https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8)[11]
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)[12]

#### Random Forest Regression model

##### Step 1: Pick at random K data points from the Training set.

##### Step 2: Build the Decision Tree associated to these K data points.

##### Step 3: Choose the number Ntree of trees you want to build and repeat STEPS 1 & 2

##### Step 4: For a new data point, make each one of your Ntree trees predict the value of Y for the data point in question, and assign the new data point the average across all of the predicted Y values.

- [basic-ensemble-learning-random-forest](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725)[13]
- [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)[14]

## Results

The app is live at https://vod6atwit-predict-price-used-toyota-app-ocyc95.streamlit.app/

## Discussion

Experimenting with various models implemented by different regression algorithms and the data was split 80/20 for testing, I found that Random Forest regression with specific number of trees provided one of the highest accuracies. I applied multiple way to [evaluate](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)[15] the model performance such as r2_score, max_error, mean_absolute_error (MAE), and mean_squared_error (MSE). Across all these evaluations, the results for the random forest regression model were the best of all model

### Multiple Linear Regression results

![Multiple Linear Regression results](./img/Multiple_Linear_Regression_results.png)

### Support Vector Regression results

![Support Vector Regression results](./img/Support_Vector_Regression_results.png)

### Decision Tree Regression results

![Decision Tree Regression results](./img/Decision_Tree_Regression_results.png)

### Random Forest Regression results

![Random Forest Regression results](./img/Random_forest_regression_results.png)

Thus, I decided to deploy the Random Forest regression model.

Some of the benefits of Random Forest Regression that I've found underline the acceptable level of success for this dataset

[Why Random Forest is My Favorite Machine Learning Model](https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706)[16]

## Summary

This sample project deploys a supervised regression model to predict the price to sell used Toyota cars on 8 features. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 96%.

The web app is designed using Streamlit and is deployed using Streamlit. The app is live at https://vod6atwit-predict-price-used-toyota-app-ocyc95.streamlit.app/.

More info about streamlit hosting is [here](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html)[17].

## References

[1] [Streamlit](https://www.streamlit.io/)

[2] [kaggle dataset](https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw)

[3] [OneHotEncoder](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)

[4] [Simple_and_multiple_linear_regression](https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression)

[5] [backward-elimination](https://www.simplilearn.com/what-is-backward-elimination-technique-in-machine-learning-article#:~:text=What%20is%20backward%20elimination%20in,is%20removed%20from%20the%20model.)

[6] [p-value_01](https://www.investopedia.com/terms/p/p-value.asp)

[7] [p-value_02](https://www.simplypsychology.org/p-value.html)

[8] [SVR](https://files.core.ac.uk/pdf/2612/81523322.pdf)

[9] [svm-kernel-functions](https://data-flair.training/blogs/svm-kernel-functions/)

[10] [entropy-information-gain-machine-learning](https://www.section.io/engineering-education/entropy-information-gain-machine-learning/)

[11] [entropy-how-decision-trees-make-decisions](https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8)

[12] [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

[13] [basic-ensemble-learning-random-forest](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725)

[14] [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

[15] [model_evaluations](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

[16] [Why Random Forest is My Favorite Machine Learning Model](https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706)

[17] [deploy_streamlit_app](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html)

<!-- [18] [GitHub Integration (Heroku GitHub Deploys)](https://devcenter.heroku.com/articles/github-integration) -->
