from .ad_campaign_optimizer import AdCampaignOptimizer
from .cohort import Cohort, CohortForecast

from sklearn.linear_model import BayesianRidge


class CampaignManager:

    """This class is used to manage the ad campaign."""

    def __init__(self, cohort_dataframes: dict):

        self.cohort_dataframes = cohort_dataframes
        self.results = {}
        self.cohorts = []

    def optimize_cohorts(self):
        for cohort_name, cohort_df in self.cohort_dataframes.items():
            print(f"Processing {cohort_name} cohort...")

            optimizer = AdCampaignOptimizer(cohort_df)
            optimizer.prepare_data()
            optimizer.fit_model(BayesianRidge())
            mse = optimizer.evaluate_model()
            optimal_price = optimizer.optimize_price_per_impression()

            self.results[cohort_name] = (optimal_price, mse)

            cohort = Cohort(cohort_name, cohort_df, optimal_price)
            print(f"Optimal Price per Impression: {optimal_price}")
            print(f"Mean Squared Error (Test Set): {mse}")
            print("\n" + "=" * 40 + "\n")
            self.cohorts.append(cohort)

    def forecast_budgets(self):
        for cohort in self.cohorts:
            cohort_forecast = CohortForecast(cohort)
            model = cohort_forecast.train_model()
            next_14_days_spend = cohort_forecast.predict_next_14_days_spend(model)
            total_budget = sum(next_14_days_spend)
            print(
                f"Total budget ${total_budget:.2f} for the next 14 days for cohort {cohort.name}"
            )

            adjusted_total_budget = cohort_forecast.calculate_adjusted_budget(
                next_14_days_spend
            )
            print(
                f"Adjusted total budget for the next 14 days: ${adjusted_total_budget:.2f}"
            )

    def run(self):
        self.optimize_cohorts()
        self.forecast_budgets()
