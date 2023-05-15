



Requirements:

1. Conda installed.


Steps:

1. Create a conda envirnoment conda env `conda env create -f environment.yml`

To see results:

Navigate to the `Example.ipynb` notebook to perform Exploratory Data Analysis (EDA) on the "marketing_data.csv" file located in the "data" folder. 
This notebook contains the following:

1. Exploration of the data.

2. Useful visualizations that help understand the data and serve as a basis to the approach to the problem.

3. To visualize the interactive dashboard build with Dash just run `python app.py` .

4. The approach to the problem for finding the optimal price per impression that returned the best converssion per session.

5. The approach to forecast for the next 14 days the adjusted budget per day.

Future work:

- Try different uses cases for optimal pricing. Demand Curve, Price Elasticity, etc.
- An approach for the pricing algorithm can be: The pricing algorithm aims to find the optimal price point (i.e., the price that maximizes profit) based on the predicted demand curve obtained from the trained model. It takes into account various factors such as the initial price (P0), the cost of the product (c), and the demand curve (predicted sales at different price points).

- Add more interactive visualizations to the dashboard.
- Transfer dashboard to a production environment port


