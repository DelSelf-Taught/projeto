import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  


class RegressionModel:    
    def __init__(self, n_samples=1000, n_features=1, noise=0.1, test_size=0.2, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()

    # Generate synthetic regression data
    def generate_data(self):
        X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features, noise=self.noise, random_state=self.random_state)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
    # Train the model
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)    
    # Evaluate the model's performance
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return mean_squared_error(y_test, y_pred)   
    # Plot the results (only for 1D feature)
    def plot_results(self, X_test, y_test): 
        if self.n_features != 1:
            raise ValueError("Plotting is only supported for 1D feature data.")
        y_pred = self.model.predict(X_test)
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.scatter(X_test, y_pred, color='red', label='Predicted')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Regression Model Results')
        plt.legend()
        plt.show()  
# Example usage
if __name__ == "__main__":
    reg_model = RegressionModel()
    X_train, X_test, y_train, y_test = reg_model.generate_data()
    reg_model.train(X_train, y_train)
    mse = reg_model.evaluate(X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
    #reg_model.plot_results(X_test, y_test)
    # trançando uma linha entre os dados
    plt.plot(X_test, y_test, 'o', label='Data points')
    plt.plot(X_test, reg_model.model.predict(X_test), 'r-', label='Regression line')
    plt.legend()
    plt.show()