# Linear Regression - Supervised Machine Learning Algorithm

## Introduction
Linear Regression is one of the most fundamental and widely used algorithms in supervised machine learning. It is used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). The goal is to find the best-fitting straight line (or hyperplane for multiple variables) that predicts the output for given input data.
It uses the equation y=mx+c where m is the slope, c is the y-intercept, x is independent and y is a dependent variable.
---

## Key Concepts

### 1. **Supervised Learning**
Linear Regression falls under supervised learning because it learns a mapping from input features to an output label based on labeled training data.

### 2. **Types of Linear Regression**
- **Simple Linear Regression**: Models the relationship between one independent variable and the dependent variable.
- **Multiple Linear Regression**: Models the relationship between two or more independent variables and the dependent variable.

### 3. **Equation of a Line**
Linear Regression models a straight-line relationship using the equation:

\[
y = \beta_0 + \beta_1x + \epsilon
\]

Where:
- \(y\) = Predicted value (dependent variable)
- \(\beta_0\) = Intercept (value of \(y\) when \(x=0\))
- \(\beta_1\) = Coefficient (slope of the line, showing how much \(y\) changes for a unit change in \(x\))
- \(x\) = Independent variable
- \(\epsilon\) = Error term (difference between actual and predicted values)

---

## How Linear Regression Works

1. **Model Assumptions**:
   - Linear relationship exists between predictors and the target variable.
   - Residuals (errors) are normally distributed.
   - Homoscedasticity: Constant variance of errors.
   - No multicollinearity in multiple linear regression.

2. **Cost Function**:
   Linear Regression uses the Mean Squared Error (MSE) as its cost function:

   \[
   MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
   \]

   Where \(y_i\) is the actual value, and \(\hat{y}_i\) is the predicted value.

3. **Optimization**:
   The algorithm minimizes the cost function (MSE) by adjusting \(\beta_0\) and \(\beta_1\) using techniques like Gradient Descent or Ordinary Least Squares (OLS).

---

## Implementation Steps

### 1. Data Preparation
   - Collect labeled data (independent variables and dependent variable).
   - Split the dataset into training and testing subsets.

### 2. Model Training
   - Fit the Linear Regression model to the training data.
   - Calculate the optimal coefficients (\(\beta_0, \beta_1, ...\)).

### 3. Model Evaluation
   - Use the testing data to evaluate the model's performance using metrics like:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - R-squared (\(R^2\)): Measures how well the model explains the variance in the data.

### 4. Prediction
   - Input new data into the trained model to make predictions.

---

## Advantages

1. Simple to understand and implement.
2. Computationally efficient for small to medium datasets.
3. Works well when the relationship between variables is linear.

---

## Limitations

1. Assumes a linear relationship, which may not hold for all datasets.
2. Sensitive to outliers, which can distort the results.
3. Prone to underfitting in complex datasets with non-linear patterns.

---

### Example Code (Python) 
#(Also see the detail implementation in given code file)

Here’s a simple implementation of Linear Regression using Python and scikit-learn:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 5, 4, 5]
})

# Split the data
X = data[['X']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
