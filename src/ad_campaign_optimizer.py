import optuna
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd


class AdCampaignOptimizer:

    """
    This class is used to optimize the price per impression for an ad campaign.

    """

    def __init__(self, df: pd.DataFrame, degree: int = 2, test_size: int = 30):
        self.df = df
        self.degree = degree
        self.test_size = test_size
        self.poly = PolynomialFeatures(degree=self.degree)
        self.log_transformer = FunctionTransformer(np.log1p, np.expm1, validate=False)

    def prepare_data(self):
        """
        This function prepares the data for the model.
        Applyes log transformation  and polynomial transformation.
        """
        self.X = self.df["price_per_impression"].values.reshape(-1, 1)
        self.y = self.df["conversion_per_session"].values.ravel()
        self.X_log = self.log_transformer.transform(self.X)
        self.X_poly = self.poly.fit_transform(self.X_log)

        TEST_SIZE = self.test_size
        self.X_train, self.X_test = (
            self.X_poly[:-TEST_SIZE, :],
            self.X_poly[-TEST_SIZE:, :],
        )
        self.y_train, self.y_test = self.y[:-TEST_SIZE], self.y[-TEST_SIZE:]

    def fit_model(self, model):
        """
        This function fits the model ususing cross validation.

        Args:
            model (_type_): model to be fitted
        """
        self.model = model
        cv_results = cross_validate(
            self.model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="neg_mean_squared_error",
            return_estimator=True,
        )
        self.model = cv_results["estimator"][np.argmax(cv_results["test_score"])]

    def evaluate_model(self):
        """
        This function evaluates the model using the test data using mean squared error.

        Returns:
            mse: mean squared error
        """
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse

    def optimize_price_per_impression(self):
        """This function optimizes the price per impression for a given model,
        using the Optuna library to conduct the optimization.
        The objective function within the optimization is designed to maximize the predicted outcome
        based on the model's prediction for a given price per impression

        Attributes:
        self.log_transformer: Preprocessing instance (e.g., LogTransformer)
        self.poly: Preprocessing instance (e.g., PolynomialFeatures)
        self.model: Model instance (e.g., BayesianRidge)
        """

        def objective_function(trial):
            """This function is used to optimize the price per impression for an ad campaign.


            Args:
                trial  (Optuna Trial object)

            Returns:
                Negative predicted outcome of the model for the given trial's price per impression
            """
            price_per_impression = trial.suggest_float("price_per_impression", 0.01, 10)
            price_array = np.array([[price_per_impression]])
            transformed_price = self.log_transformer.transform(price_array)
            price_poly = self.poly.transform(transformed_price)
            return -self.model.predict(price_poly).item()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective_function, n_trials=50)
        return study.best_params["price_per_impression"]
