import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🛒 Store Item Demand Forecasting App")

# Create tabs
tab1, tab2 = st.tabs(["📄 Project Summary", "📈 Forecasting App"])

with tab1:
    st.header("Project Overview")
    st.markdown("""
The Store Item Demand Forecasting App is designed to predict future sales for individual store-item combinations using advanced time series forecasting techniques. By leveraging historical sales data, the app empowers businesses to optimize inventory management, streamline supply chain operations, and make data-driven decisions.

🔍 Problem Statement
Retail businesses face the challenge of accurately forecasting demand at the store-item level to avoid overstocking or stockouts. The goal is to predict 90-day future sales for each store-item pair, capturing trends, seasonality, and volatility to improve operational efficiency.

📦 Dataset Overview
The dataset used in this project contains daily sales records for multiple stores and items. Key features include:

Columns:
date: Date of the sales record (format: DD-MM-YYYY)
store: Store identifier
item: Item identifier
sales: Number of units sold


Time Span: Multi-year daily data
Granularity: Store-item level sales

The dataset is preprocessed to extract additional temporal features such as day, month, year, and dayofweek for enhanced analysis.

🧠 Methodology
The app employs a robust pipeline to preprocess data, analyze patterns, and generate forecasts. The methodology includes:

Data Preprocessing:

Convert date to datetime format and set as index
Filter data by user-selected store and item
Extract temporal features (day, month, year, day of week)


Exploratory Data Analysis (EDA):

Visualize sales trends over time using interactive Plotly charts
Perform seasonal decomposition to identify trend, seasonality, and residual components
Compute rolling statistics (mean and standard deviation) to assess stationarity
Conduct Augmented Dickey-Fuller (ADF) test to confirm stationarity of the time series


Forecasting Models: A variety of time series models are implemented to capture different aspects of the data:

ARIMA: Captures autoregressive and moving average components
SARIMA: Extends ARIMA to model seasonality
Exponential Smoothing: Accounts for trend and seasonality with additive components
ARCH/GARCH: Models volatility in sales data
LSTM (Long Short-Term Memory): A deep learning approach to capture complex temporal dependencies


Evaluation:

Forecasts are evaluated over a 90-day test period
Visual comparisons of predicted vs. actual sales
Confidence intervals provided for SARIMA forecasts




🛠️ Technical Stack

Frontend: Streamlit for an interactive web interface
Data Processing: Pandas, NumPy
Visualization: Plotly for dynamic, interactive charts
Statistical Modeling: Statsmodels (ARIMA, SARIMA, Exponential Smoothing), ARCH
Deep Learning: TensorFlow/Keras for LSTM
Stationarity Testing: Statsmodels (ADF test)


📊 Key Features

Interactive Filters: Select specific store and item combinations via sidebar
Data Visualization:
Line charts for sales trends
Seasonal decomposition plots (observed, trend, seasonal, residual)
Rolling mean and standard deviation plots


Stationarity Analysis: ADF test results and differenced series visualization
Multiple Forecasting Models:
ARIMA, SARIMA, Exponential Smoothing, ARCH/GARCH, and LSTM
90-day forecasts with confidence intervals (for SARIMA)


User-Friendly Interface: Tabbed layout for project summary and forecasting app


📈 Outputs

90-Day Sales Forecast: Predicted sales for the selected store-item pair
Confidence Intervals: Uncertainty bounds for SARIMA forecasts
Visual Insights:
Historical sales trends
Decomposition of time series components
Forecast vs. actual sales comparisons


Volatility Analysis: GARCH-based variance forecasts


🚀 Business Impact

Inventory Optimization: Accurate demand forecasts reduce overstocking and stockouts
Supply Chain Efficiency: Improved planning for procurement and logistics
Cost Savings: Minimized waste and storage costs
Scalability: The app can be extended to handle additional stores, items, or forecasting horizons


🔮 Future Enhancements

Model Ensemble: Combine predictions from multiple models for improved accuracy
Hyperparameter Tuning: Automate optimization of model parameters
External Features: Incorporate exogenous variables (e.g., holidays, promotions)
Real-Time Updates: Integrate live data feeds for dynamic forecasting
Cross-Store Analysis: Identify patterns across multiple stores or items


📝 Conclusion
The Store Item Demand Forecasting App provides a powerful, user-friendly solution for predicting retail sales at the store-item level. By combining statistical and deep learning models with interactive visualizations, it equips businesses with the insights needed to make informed decisions and stay ahead in a competitive market.

    """)

with tab2:
    uploaded_file = st.sidebar.file_uploader("Upload dataset (train.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload the dataset used in the original notebook to proceed.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

    def expand_df(df):
        data = df.copy()
        data['day'] = data.date.dt.day
        data['month'] = data.date.dt.month
        data['year'] = data.date.dt.year
        data['dayofweek'] = data.date.dt.dayofweek
        return data

    df = expand_df(df)

    st.sidebar.subheader("Filter Data")
    store = st.sidebar.selectbox("Store", sorted(df['store'].unique()))
    item = st.sidebar.selectbox("Item", sorted(df['item'].unique()))
    filtered_df = df[(df['store'] == store) & (df['item'] == item)].copy()
    filtered_df.set_index('date', inplace=True)
    filtered_df.sort_index(inplace=True)

    st.subheader(f"Data Preview: Store {store}, Item {item}")
    st.write(filtered_df.head())

    st.subheader("Sales Over Time")
    fig1 = px.line(filtered_df, x=filtered_df.index, y='sales', title="Sales Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Seasonal Decomposition")
    result = sm.tsa.seasonal_decompose(filtered_df['sales'], model='additive', period=365)
    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig2.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Observed"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name="Trend"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig2.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name="Residual"), row=4, col=1)
    fig2.update_layout(height=800, title="Seasonal Decomposition", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    def plot_rolling_stats_plotly(ts, window=12):
        rolmean = ts.rolling(window).mean()
        rolstd = ts.rolling(window).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts, name='Original'))
        fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean, name='Rolling Mean'))
        fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd, name='Rolling Std'))
        fig.update_layout(title="Rolling Mean & Std Dev", height=400)
        return fig

    st.subheader("Rolling Statistics for Stationarity")
    st.plotly_chart(plot_rolling_stats_plotly(filtered_df['sales']), use_container_width=True)

    def dickey_fuller_test(timeseries):
        dftest = adfuller(timeseries, autolag='AIC')
        return {
            'Test Statistic': dftest[0],
            'p-value': dftest[1],
            'Lags Used': dftest[2],
            'Number of Observations': dftest[3],
            'Critical Values': dftest[4]
        }

    st.write("Dickey-Fuller Test:")
    st.write(dickey_fuller_test(filtered_df['sales']))

    first_diff = filtered_df['sales'].diff().dropna()
    st.subheader("Differenced Series")
    st.plotly_chart(plot_rolling_stats_plotly(first_diff), use_container_width=True)
    st.write(dickey_fuller_test(first_diff))

    # Forecasting models
    st.subheader("Forecasting Models")

    train = filtered_df['sales'][:-90]
    test = filtered_df['sales'][-90:]

    if st.button("Run ARIMA Forecast"):
        arima_model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = arima_model.forecast(steps=90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig.add_trace(go.Scatter(x=test.index, y=forecast, name="ARIMA Forecast"))
        fig.update_layout(title="ARIMA Forecast")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Run Exponential Smoothing Forecast"):
        exp_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365).fit()
        forecast = exp_model.forecast(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig.add_trace(go.Scatter(x=test.index, y=forecast, name="Exponential Smoothing Forecast"))
        fig.update_layout(title="Exponential Smoothing Forecast")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Run ARCH/GARCH Forecast"):
        returns = 100 * filtered_df['sales'].pct_change().dropna()
        arch_mod = arch_model(returns, vol='Garch', p=1, q=1)
        res = arch_mod.fit(disp='off')
        forecast = res.forecast(horizon=90)
        vol = forecast.variance.values[-1, :]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 91)), y=vol, name="Forecasted Variance"))
        fig.update_layout(title="GARCH Forecast of Volatility")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecasting with SARIMA")
    if st.button("Run SARIMA Forecast"):
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit()
        forecast = results.get_forecast(steps=90)
        forecast_df = forecast.predicted_mean
        conf_int = forecast.conf_int()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig3.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig3.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df, name="Forecast", line=dict(color='green')))
        fig3.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line_color='lightgrey', name='Lower CI'))
        fig3.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper CI'))
        fig3.update_layout(title="SARIMA Forecast", height=500)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Forecasting with LSTM")
    def create_lstm_dataset(series, look_back=30):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(filtered_df[['sales']].values)
    X_lstm, y_lstm = create_lstm_dataset(scaled_data)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)
    X_train_lstm = X_lstm[:-90]
    y_train_lstm = y_lstm[:-90]
    X_test_lstm = X_lstm[-90:]
    y_test_lstm = y_lstm[-90:]

    if st.button("Run LSTM Forecast"):
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(30, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
        predictions = model.predict(X_test_lstm)
        predictions_rescaled = scaler.inverse_transform(predictions)
        forecast_dates = filtered_df.index[-90:]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=filtered_df.index[-180:], y=filtered_df['sales'].values[-180:], name="Actual"))
        fig4.add_trace(go.Scatter(x=forecast_dates, y=predictions_rescaled.flatten(), name="LSTM Forecast", line=dict(color='orange')))
        fig4.update_layout(title="LSTM Forecast vs Actual", height=500)
        st.plotly_chart(fig4, use_container_width=True)
