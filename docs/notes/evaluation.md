# Evaluation Metrics

## RMSE

**Root Mean Square Error (RMSE)** is used to measure the real value and the observed value, its formula is given as follows:
$$
RMSE(X,F)=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(F(x_i)-y_i)^2}
$$
 where $X$ denotes the dataset, $F$ denotes the specific function.

In `Beta-Recsys`, you can calculate the `RMSE` as follows:

``` python
from beta_rec.utils.evaluation import rmse
rmse(rating_true, rating_pred)
```

## MAE

**Mean Absolute Error (MAE)** is the average value of absolute value, it could reflect the real situation of observed error, its formula is given as follows:
$$
MAE(X, F)=\frac{1}{N}\sum_{i=1}^{N}|h(x_i)-y_i|
$$
 where $X$ denotes the dataset, $F$ denotes the specific function.

## R-square

**R-Squared ($R^2 $)** is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.  The formula for R-Squared is :
$$
R^2 = 1 - \frac{Unexplained \quad Variation}{Total \quad Variation}
$$

## Explained Variance

## AUC

**Area Under Curve (AUC)** is used in classification analysis in order to determine which of the used models predicts the classes best.

## logloss

log loss is defined by the cross-entropy loss function, and its formula is as follows:
$$
LogLoss=-\frac{1}{N}\sum_{n=1}^{N}[y_n\log\hat{y_n}+(1-y_n)\log(1-\hat{y_n})]
$$
where $\hat{y_i}$ represents whether the predicted result is positive.