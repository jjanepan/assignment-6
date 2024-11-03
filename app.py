from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def create_dataset(N, mu, sigma):
    """Generates random X values and Y values with normal error."""
    X = np.random.uniform(0, 1, N)
    Y = X + np.random.normal(mu, sigma, N)
    return X, Y

def fit_linear_regression(X, Y):
    """Fits a linear regression model and returns the slope and intercept."""
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    return model.coef_[0], model.intercept_

def save_scatter_plot(X, Y, slope, intercept, filepath="static/plot1.png"):
    """Generates and saves a scatter plot with a regression line."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')

    X_sorted = np.sort(X)
    Y_pred = slope * X_sorted + intercept
    plt.plot(X_sorted, Y_pred, color='red', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def save_histogram(slopes, intercepts, slope, intercept, filepath="static/plot2.png"):
    """Generates and saves a histogram for slopes and intercepts from simulations."""
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")

    plt.axvline(slope, color="blue", linestyle="--", label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", label=f"Intercept: {intercept:.2f}")
    
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Slopes and Intercepts")
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def run_simulations(N, mu, sigma2, S):
    """Runs multiple simulations to gather slopes and intercepts, and calculates extreme values."""
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim, Y_sim = create_dataset(N, mu, np.sqrt(sigma2))
        sim_slope, sim_intercept = fit_linear_regression(X_sim, Y_sim)
        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    return slopes, intercepts

def calculate_extreme_values(slopes, intercepts, slope, intercept):
    """Calculates the proportion of more extreme slope and intercept values."""
    slope_extreme_count = sum(s > slope for s in slopes) / len(slopes)
    intercept_extreme_count = sum(i < intercept for i in intercepts) / len(intercepts)
    return slope_extreme_count, intercept_extreme_count

def generate_plots(N, mu, sigma2, S):
    """Generates all required plots and calculates results."""
    # Step 1: Generate initial dataset and fit regression model
    X, Y = create_dataset(N, mu, np.sqrt(sigma2))
    slope, intercept = fit_linear_regression(X, Y)

    # Step 2: Save scatter plot with regression line
    save_scatter_plot(X, Y, slope, intercept)

    # Step 3: Run simulations to collect slopes and intercepts
    slopes, intercepts = run_simulations(N, mu, sigma2, S)

    # Step 4: Save histogram of simulated slopes and intercepts
    save_histogram(slopes, intercepts, slope, intercept)

    # Step 5: Calculate proportions of more extreme slopes and intercepts
    slope_extreme, intercept_extreme = calculate_extreme_values(slopes, intercepts, slope, intercept)

    return "static/plot1.png", "static/plot2.png", slope_extreme, intercept_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user input values
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        # Render with plots and results
        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    # Initial page load
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)