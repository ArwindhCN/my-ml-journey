import numpy as np

def sigmoid(z):
    # This is the "Governor" that squashes any number between 0 and 1
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=5000):
    m = 0.0
    b = 0.0
    n = len(X)

    for i in range(iterations):
        # 1. Linear combination (The straight line)
        z = m * X + b
        
        # 2. Apply Sigmoid (The S-Curve fix)
        y_pred = sigmoid(z)
        
        # 3. Calculate Gradients (Notice: The math looks same as Linear!)
        dm = (1/n) * np.sum(X * (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        
        # 4. Update
        m = m - (learning_rate * dm)
        b = b - (learning_rate * db)
        
        if i % 1000 == 0:
            # We calculate "Log Loss" here to see how well we are doing
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            print(f"Iteration {i}: Loss = {loss:.4f}")

    return m, b

# --- Testing the Fix ---
# X = Hours spent looking at a URL, y = 1 (Phishing) or 0 (Safe)
X_test = np.array([1, 2, 3, 7, 8, 9])
y_test = np.array([0, 0, 0, 1, 1, 1]) 

m, b = logistic_regression(X_test, y_test)

# Test a new URL that was seen for 10 hours
new_z = m * 10 + b
probability = sigmoid(new_z)
print(f"\nProbability of Phishing for 10 hours: {probability:.4f}") 
# This will be close to 1.0 (100%), but NEVER exceed it.
