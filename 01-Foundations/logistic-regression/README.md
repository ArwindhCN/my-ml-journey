# Logistic Regression: Binary Classification

Logistic Regression is used to predict the probability of a categorical outcome (e.g., 1 for Phishing, 0 for Safe). It solves the "straight line" limitation of Linear Regression by using the Sigmoid function.

## 1. The Sigmoid Function (The "Governor")
To ensure our predictions ($\hat{y}$) stay between 0 and 1, we pass our linear equation $z = mx + b$ through the Sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$



## 2. The Loss Function: Log Loss (Binary Cross-Entropy)
In classification, we don't use Mean Squared Error. Instead, we use **Log Loss**. Loss is a mathematical penalty: the further the prediction is from the actual label, the higher the penalty.

$$J = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

### How it works:
* **If the actual label is 1:** The loss increases exponentially as the prediction ($\hat{y}$) approaches 0.
* **If the actual label is 0:** The loss increases exponentially as the prediction ($\hat{y}$) approaches 1.



## 3. The Gradient Descent Update
Interestingly, the partial derivatives for Log Loss combined with the Sigmoid function result in the same update rule as Linear Regression:

* **dm:** $\frac{1}{n} \sum x_i(\hat{y}_i - y_i)$
* **db:** $\frac{1}{n} \sum (\hat{y}_i - y_i)$

This allows us to reuse our Gradient Descent engine with only a change to how $\hat{y}$ is calculated.

