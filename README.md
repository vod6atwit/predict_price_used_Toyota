## Introduction

The objective of this project is to train, build, test and select the best regression model to predict the price to sell used Toyota cars, considering their year, transmission, mileage, fuel type, tax, mpg, and engine size. I am using the Heroku platform as a service (PaaS) for the automatic deployment of a machine-learning web app.

I wanted to ease the web app development and [Streamlit](https://www.streamlit.io/) made this possible [1]. It is an open-source library that focuses on data science and ML web app development.

## Selection of Data

TODO: edit the links

All the data preprocessing steps are conducted using a Jupyter Notebook and is available [here](https://github.com/memoatwit/dsexample/blob/master/Insurance%20-%20Model%20Training%20Notebook.ipynb).

All the models training, building, testing, and selecting are conducted using a Jupyter Notebook and is available [here](https://github.com/memoatwit/dsexample/blob/master/Insurance%20-%20Model%20Training%20Notebook.ipynb).

The data has over 6500 samples with 8 independent/feature variables: model, year, transmission, mileage, fuelType, tax, mpg, and engineSize with 1 dependent variable: Price
The objective is to indicate the price to sell used Toyota cars.
The dataset can found online at [kaggle](https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw)[2].

TODO: edit after finishing the code

Data preview:
![data overview screenshot](./img/data_overview_01.png)

![data info screenshot](./img/data_overview_02.png)

Note that data has categorical features in 3 cols: model, transmission, and fuelType.

I used OneHotEncoder/ColumnTransformer on these features and kept the rest of the features as is. When using a Random Forest Regression model, the r2_score always over 95%.

We then create a pipeline for automating the process for new data. I also experimented with different imputer strategies for missing data and added second-degree polynomial features for both the numeric and categorical data. The shown values are obtained by performing a grid search over these arguments.
Pipeline preview:
![pipeline screenshot](./pipeline.png)

I finally saved the model via joblib to be used for predictions by the web app.

## Methods

Tools:

- NumPy, SciPy, Pandas, and Scikit-learn for data analysis and inference
- Streamlit (st) for web app design
- GitHub and Heroku for web app deployment and hosting/version control
- VS Code as IDE

Inference methods used with Scikit:

- modules: metrics, compose, preprocessing, model_selection
- classes: ColumnTransformer, OneHotEncoder, StandardScaler, train_test_split, r2_score, max_error, mean_absolute_error, mean_squared_error
- linear regression model
- Vector regression model
- Decision tree regression model
- Random Forest regression model

- Pipeline to tie it all together

## Results

The app is live at https://ds-example.herokuapp.com/
It allows for online and batch processing as designed by the pycaret post:

- Online: User inputs each feature manually for predicting a single insurance cost
  ![online screenshot](./online.png)
- Batch: It allows the user to upload a CSV file with the 6 features for predicting many instances at once.
  - An [X_test.csv](./X_test.csv) is provided as a batch processing sample. Corresponding insurance prices are available at [y_test.csv](./y_test.csv)
    ![batch screenshot](./batch.png)

I am not adding any visualizations to this example, though st supports it. Couple good examples are [here](https://share.streamlit.io/tylerjrichards/book_reco/books.py) and [here](https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/)

## Discussion

Experimenting with various models implemented by different regression algorithms and the data was split 80/20 for testing, I found that Random Forest regression with specific number of trees provided one of the highest accuracies. I applied multiple way to evaluate the model performance such as r2_score, max_error, mean_absolute_error (MAE), and mean_squared_error (MSE). Across all these evaluations, the results for the random forest regression model were the best of all model

![Multiple Linear Regression results](./img/Multiple_Linear_Regression_results.png)
![Support Vector Regression results](./img/Support_Vector_Regression_results.png)
![Decision Tree Regression results](./img/Decision_Tree_Regression_results.png)
![Random Forest Regression results](./img/Random_forest_regression_results.png)

Thus, I decided the deploy the pipelined Random Forest regression model.

Some of the benefits of Random Forest Regression that I've found underline the acceptable level of success for this dataset

![Why Random Forest is My Favorite Machine Learning Model](https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706)

TODO: edit this line

One unexpected challenge was the free storage capacity offered by Heroku. I experimented with various versions of the libraries listed in `requirements.txt` to achieve a reasonable memory footprint. While I couldn't include the latest pycaret library due to its size, the current setup does include TensorFlow 2.3.1 (even though not utilized by this sample project) to illustrate how much can be done in Heroku's free tier:

```
Warning: Your slug size (326 MB) exceeds our soft limit (300 MB) which may affect boot time.
```

## Summary

This sample project deploys a supervised regression model to predict insurance costs based on 6 features. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 73%.

The web app is designed using Streamlit, and can do online and batch processing, and is deployed using Heroku and Streamlit. The Heroku app is live at https://ds-example.herokuapp.com/.

Streamlit is starting to offer free hosting as well. The same repo is also deployed at [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/memoatwit/dsexample/app.py)  
More info about st hosting is [here](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html).

## References

[1] [GitHub Integration (Heroku GitHub Deploys)](https://devcenter.heroku.com/articles/github-integration)

[2] [Streamlit](https://www.streamlit.io/)

[3] [The pycaret post](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)

[4] [Insurance dataset: git](https://github.com/stedy/Machine-Learning-with-R-datasets)

[5] [Insurance dataset: kaggle](https://www.kaggle.com/mirichoi0218/insurance)
