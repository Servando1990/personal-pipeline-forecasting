from prophet import Prophet
import pandas as pd


class Cohort:
    def __init__(self, name: str, dataframe: pd.DataFrame, optimal_price: float):
        self.name = name
        self.dataframe = dataframe
        self.optimal_price = optimal_price


class CohortForecast:
    """This class is used to forecast the spend for a cohort."""

    def __init__(self, cohort: Cohort):
        self.cohort = cohort

    def daily_spend(self) -> pd.DataFrame:
        """This function calculates the daily spend for a cohort.
        Returns: Daily spend for a cohort"""

        daily_spend = (
            self.cohort.dataframe.groupby("date")["total_spend"].sum().reset_index()
        )
        daily_spend.rename(columns={"total_spend": "daily_spend"}, inplace=True)
        daily_spend.columns = ["ds", "y"]
        return daily_spend

    def train_model(self) -> Prophet:
        """This function trains a Prophet model for a cohort.
        Returns: Trained Prophet model"""
        model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=False,
            weekly_seasonality=False,
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.fit(self.daily_spend())
        return model

    def predict_next_14_days_spend(self, model: Prophet) -> pd.DataFrame:
        """This function predicts the spend for the next 14 days.
        Args:
            model (Prophet): trained Prophet model
            Returns: Forecast for the next 14 days"""
        future = model.make_future_dataframe(periods=14)
        forecast = model.predict(future)
        return forecast.loc[-14:, "yhat"].values

    def calculate_adjusted_budget(self, next_14_days_spend: pd.DataFrame) -> float:
        """This function calculates the adjusted budget for the next 14 days.
        Args:
            next_14_days_spend (pd.DataFrame): spend for the next 14 days
            Returns: Adjusted budget for the next 14 days
        """
        historical_average_price_per_impression = self.cohort.dataframe[
            "price_per_impression"
        ].mean()
        adjustment_factor = (
            self.cohort.optimal_price / historical_average_price_per_impression
        )
        adjusted_next_14_days_spend = next_14_days_spend * adjustment_factor
        return sum(adjusted_next_14_days_spend)
