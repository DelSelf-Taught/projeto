from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MachineLearningModel:
    def __init__(self, n_samples=1000, n_features=20, n_classes=2, test_size=0.2, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)

    def generate_data(self):
        X, y = make_blobs(n_samples=self.n_samples, centers=self.n_classes, n_features=self.n_features, random_state=self.random_state)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
if __name__ == "__main__":  
    ml_model = MachineLearningModel()
    X_train, X_test, y_train, y_test = ml_model.generate_data()
    ml_model.train(X_train, y_train)
    accuracy = ml_model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")