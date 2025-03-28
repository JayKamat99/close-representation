import numpy as np
import scipy.optimize as opt

def fit_curve(data, degree):
    # Extract x and y coordinates
    x = data[:, 0]
    y = data[:, 1]

    # Fit a cubic polynomial
    coefficients = np.polyfit(x, y, degree)

    # Create a polynomial function
    p = np.poly1d(coefficients)

    # Generate smooth x values for plotting the curve
    # x_fit = np.linspace(min(x), max(x), 100)
    x_fit = np.linspace(0, 1, 100)
    y_fit = p(x_fit)
    
    return x_fit, y_fit

def fit_poly_curve(data, degree):
    # Extract x and y coordinates
    x = data[:, 0]
    y = data[:, 1]%(2*np.pi)

    # Define polynomial model
    def poly_model(coeffs, x):
        """Compute y values for a polynomial with given coefficients."""
        return sum(c * x**i for i, c in enumerate(coeffs))

    # Define custom loss function (absolute error)
    def circular_loss(coeffs, x, y):
        """Compute custom loss (L1 loss in this case)."""
        y_pred = poly_model(coeffs, x)%(2*np.pi)
        return np.sum(diff(y, y_pred)**2)  # circular L2 loss

    # Initial guess: Random or zeros (avoiding np.polyfit)
    initial_guess = np.zeros(degree+1)  # degree+1 coefficients for a polynomial of degree
    initial_guess[0] = y[-1]%(2*np.pi)

    # Optimize using scipy.optimize.minimize
    result = opt.minimize(circular_loss, initial_guess, args=(x, y))

    # Extract optimized coefficients
    optimized_coeffs = result.x
    p_custom = np.poly1d(optimized_coeffs[::-1])  # Reverse order for np.poly1d

    # Generate smooth x values for plotting
    x_fit = np.linspace(0, 1, 100)
    y_fit = poly_model(optimized_coeffs, x_fit)
    
    return [x_fit, y_fit]

def diff(theta1, theta2):
    abs_diff = abs(theta1 -theta2)
    return np.minimum(abs_diff, 2*np.pi-abs_diff)

# Define a custom distance function on circle
def circular_euclidean_distance(x, y):
    r1, theta1  = x
    r2, theta2 = y
    return np.sqrt((r1-r2)**2 + (diff(theta1, theta2))**2)