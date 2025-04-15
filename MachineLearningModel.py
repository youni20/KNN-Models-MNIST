from abc import ABC, abstractmethod
import numpy as np
from collections import Counter

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass


    def _euclidean_distance(self, x: float, y: float) -> float:
        sum = np.sum((x - y)**2)
        distance = np.sqrt(sum)
        return distance


    def normalize(self, x: np.array) -> np.array:
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_std = np.where(x_std==0, 1.0, x_std)
        normalized_x = (x - x_mean) / x_std
        

        return normalized_x
    

class KNNRegressionModel(MachineLearningModel):
    """
    Class for KNN regression model.
    """

    def __init__(self, k: int):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y) -> None:
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """

        self.X_train = X
        self.y_train = y
    

    def predict(self, X: np.array) -> np.array: 
        """
        Make predictions on new data by finding k-nearest neighbors in training data.
        
        Parameters:
        X (array-like): Features of the new data (can be single point or multiple points)
        
        Returns:
        predictions (array-like): Predicted values
        """
        k = self.k
        predictions = []
        X_train_norm = self.normalize(self.X_train)
        for test_point in X:
            test_point_norm = (test_point - np.mean(self.X_train)) / np.std(self.X_train)
            distances = []
            for train_point in X_train_norm:
                distance = self._euclidean_distance(test_point_norm, train_point)
                distances.append(distance)
            
            nearest_indices = np.argsort(distances)[:k]
            nearest_y = self.y_train[nearest_indices]
            
            # Prediction is mean of teh neighbors y values
            prediction = np.mean(nearest_y)
            predictions.append(prediction)

        return np.array(predictions)


    def evaluate(self, y_true: np.array, y_predicted: np.array) -> float:
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the Mean Squared Error (MSE) between the true and predicted values.
        The MSE is calculated as the average of the squared differences between the true and predicted values.        

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        squared_errors = []

        if y_true.shape != y_predicted.shape:
            raise ValueError("Input arrays must have the same shape")
        else:
            for i in range(len(y_predicted)):
                squared_errors.append((y_true[i] - y_predicted[i]) ** 2)

        return np.mean(squared_errors)


class KNNClassificationModel(MachineLearningModel):
    """
    Class for KNN classification model.
    """

    def __init__(self, k: int):
        """
        Initialize the model with the specified instructions.
        Parameters: k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X: np.array, y: np.array) -> None:
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:    
        None
        """
        
        self.X_train = X
        self.y_train = y
        

    def predict(self, X: np.array) -> np.array:
        """
        Make predictions on new data.
        The predictions are made by taking the mode (majority) of the target variable of the k nearest neighbors.
        
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """    
        predictions = []
        mean = np.mean(self.X_train, axis=0)
        std = np.std(self.X_train, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero

        X_train_norm = (self.X_train - mean) / std
        
        for test_point in X:
            test_point_norm = (test_point - mean) / std
            distances = np.sqrt(np.sum((X_train_norm - test_point_norm)**2, axis=1))
            
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)


    def evaluate(self, y_true: np.array, y_predicted: np.array) -> float:
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the total number of correct predictions only.
        Do not use any other evaluation metric.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        correct = 0

        if len(y_true) != len(y_predicted):
            raise ValueError("The true labels and predicted labels arrays must have the same length.")

        correct = np.sum(y_true == y_predicted)
        return correct