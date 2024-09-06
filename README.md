# 🐂 Bull's AI: Algorithmic Stock Prediction

![Bull's AI Logo](BullsAI.png)

## 📊 Empowering Investors with Machine Learning

**[Try Bull's AI Now](https://bullai.streamlit.app)**

Bull's AI is a cutting-edge stock price prediction platform that harnesses the power of machine learning to assist investors in making informed, data-driven decisions. By combining advanced algorithms with real-time market data, we provide a comprehensive tool for stock analysis and forecasting.

## 🌟 Key Features

- **Live Market Data**: Access up-to-the-minute stock prices and crucial financial metrics
- **Dynamic Visualization**: Explore historical trends and price predictions through interactive charts
- **ARIMA Forecasting**: Leverage sophisticated statistical models for accurate price projections
- **In-Depth Stock Profiles**: Obtain comprehensive information on any publicly traded company
- **Intuitive User Experience**: Seamlessly navigate and analyze stocks with our user-friendly interface

## 🛠️ Technology Stack

- **[Streamlit](https://streamlit.io/)**: Powering our interactive and responsive web interface
- **[YFinance](https://pypi.org/project/yfinance/)**: Fetching real-time and historical financial data from Yahoo Finance
- **[StatsModels](https://www.statsmodels.org/)**: Implementing robust ARIMA time series forecasting
- **[Plotly](https://plotly.com/)**: Generating dynamic and informative financial visualizations
- **[Pandas](https://pandas.pydata.org/)**: Enabling efficient data manipulation and analysis

## 🧠 Machine Learning Models

Our stock prediction engine primarily utilizes:

1. **AutoRegressive (AR) Model**: 
   - A core component of the ARIMA family
   - Implemented via `statsmodels.tsa.ar_model.AutoReg`
   - Utilizes historical price data to forecast future trends

2. **Advanced Time Series Forecasting**:
   - Employs data splitting for training and testing
   - Generates predictions on the test set and projects 90 days into the future

## 🚀 Quick Start Guide

### System Requirements
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bulls-ai.git
   ```
2. Navigate to the project folder:
   ```
   cd bulls-ai
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Launch the Streamlit app:
   ```
   streamlit run 🏠Home.py
   ```
5. Access the application at `http://localhost:8501` in your web browser

## 📚 Usage Guide

1. **Stock Information**: Choose a stock to view detailed company data and key performance indicators
2. **Price Prediction**: Select a stock, time frame, and interval to generate price forecasts
3. **Market Analysis**: Utilize our interactive charts to identify trends and make strategic decisions

## ⚠️ Important Considerations

- Our AR model assumes that future prices correlate with historical data, which may not always hold true in volatile markets
- The current model does not account for external factors such as breaking news or macroeconomic shifts
- Stock market prediction is inherently complex and uncertain; use these forecasts as one tool among many in your investment strategy

## 🔮 Roadmap

We're committed to continuous improvement. Future enhancements may include:

- Integration of more sophisticated ARIMA or SARIMA models for nuanced time series analysis
- Incorporation of advanced machine learning algorithms like Random Forests or Gradient Boosting Machines
- Implementation of deep learning architectures such as LSTM networks for improved pattern recognition
- Integration of diverse external data sources for more comprehensive market analysis

## 📄 Licensing

This project is distributed under the MIT License. For full details, please refer to the [LICENSE.md](LICENSE.md) file.

## 👨‍💻 Meet the Team

- [Sahil Bhoite](https://www.linkedin.com/in/sahil-bhoite/): AI & Tech Enthusiast | LLMs | Java Developer | Machine Learning & Data Science Specialist | Python | Pune
- [Maheshwari Jadhav](https://www.linkedin.com/in/maheshwari-jadhav/): Java | Python | AI/ML | Generative AI | Frontend Development | MITAOE'25

We welcome contributions and feedback from the community. Together, let's revolutionize stock market analysis!
