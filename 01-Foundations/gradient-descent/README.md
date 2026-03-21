# Gradient Descent: Mathematical Optimization

Gradient Descent is the primary optimization engine used to minimize the Cost Function of a machine learning model. It iteratively adjusts the model parameters (weights and bias) to find the local minimum of the error surface.

## 1. The Cost Function (Mean Squared Error)
To measure how "wrong" our line of best fit is, we use the average of the squared differences between predicted values and actual values:

$$J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

## 2. The Mechanics (Partial Derivatives)
To move "downhill" toward the minimum error, we calculate the gradient (slope) of the Cost Function with respect to each parameter. This requires the Chain Rule:

* **Weight Gradient (dm):** $$\frac{\partial J}{\partial m} = \frac{2}{n} \sum -x_i(y_i - \hat{y}_i)$$
* **Bias Gradient (db):** $$\frac{\partial J}{\partial b} = \frac{2}{n} \sum -(y_i - \hat{y}_i)$$

## 3. The Update Rule
In every iteration, we update the parameters by stepping in the opposite direction of the gradient, scaled by a Learning Rate (Alpha):

$$m = m - \alpha \cdot dm$$
$$b = b - \alpha \cdot db$$

## 4. Implementation Details
The Python implementation in this folder uses Vectorized NumPy operations to calculate these gradients across the entire dataset simultaneously, bypassing the need for slow Python `for` loops.
