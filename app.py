import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
from textblob import TextBlob

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Helper Functions
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df, stock.info
    except:
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build LSTM neural network"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock_price(df, days_ahead=30):
    """Predict stock prices using LSTM"""
    with st.spinner('Training AI model... This may take a moment.'):
        close_prices = df['Close'].values
        X, y, scaler = prepare_lstm_data(close_prices)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train model
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Make predictions
        predictions = []
        last_sequence = close_prices[-60:]
        
        for _ in range(days_ahead):
            scaled_seq = scaler.transform(last_sequence.reshape(-1, 1))
            X_pred = scaled_seq.reshape((1, 60, 1))
            pred = model.predict(X_pred, verbose=0)
            pred_price = scaler.inverse_transform(pred)[0][0]
            predictions.append(pred_price)
            last_sequence = np.append(last_sequence[1:], pred_price)
        
        return predictions

def get_stock_news(ticker):
    """Fetch recent news for stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        return news
    except:
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def ai_chatbot_response(user_query, ticker_data=None):
    """Simple AI chatbot for stock queries"""
    query_lower = user_query.lower()
    
    if 'price' in query_lower or 'cost' in query_lower:
        if ticker_data:
            return f"The current price is ${ticker_data['Close'].iloc[-1]:.2f}"
        return "Please select a stock first to get price information."
    
    elif 'high' in query_lower:
        if ticker_data:
            return f"The 52-week high is ${ticker_data['Close'].max():.2f}"
        return "Please select a stock first."
    
    elif 'low' in query_lower:
        if ticker_data:
            return f"The 52-week low is ${ticker_data['Close'].min():.2f}"
        return "Please select a stock first."
    
    elif 'buy' in query_lower or 'sell' in query_lower:
        return "I cannot provide financial advice. Please consult with a financial advisor."
    
    elif 'predict' in query_lower or 'forecast' in query_lower:
        return "Use the 'ML Prediction' tab to see AI-powered price predictions based on historical data."
    
    else:
        return "I can help you with stock prices, technical analysis, and predictions. What would you like to know?"

# Main App
st.title("📈 AI-Powered Stock Prediction Platform")
st.markdown("*Advanced Machine Learning & Technical Analysis for Smart Investing*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Stock selection
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    st.markdown("---")
    st.subheader("📊 Popular Stocks")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("AAPL"): ticker = "AAPL"
        if st.button("GOOGL"): ticker = "GOOGL"
        if st.button("TSLA"): ticker = "TSLA"
    with col2:
        if st.button("MSFT"): ticker = "MSFT"
        if st.button("AMZN"): ticker = "AMZN"
        if st.button("NVDA"): ticker = "NVDA"

# Fetch data
df, info = get_stock_data(ticker, period)

if df is not None and not df.empty:
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
    with col2:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    with col3:
        st.metric("52W High", f"${df['Close'].max():.2f}")
    with col4:
        st.metric("52W Low", f"${df['Close'].min():.2f}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Chart", "🤖 ML Prediction", "📈 Technical Analysis", "💬 AI Chatbot", "📰 News & Sentiment"])
    
    # Tab 1: Interactive Chart
    with tab1:
        st.subheader(f"{ticker} Stock Price Chart")
        
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Price"
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], name="Close Price",
                line=dict(color='#2962FF', width=2)
            ), row=1, col=1)
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20",
                                line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50",
                                line=dict(color='red', width=1)), row=1, col=1)
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                            marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: ML Prediction
    with tab2:
        st.subheader("🤖 LSTM Neural Network Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            days_ahead = st.slider("Prediction Period (days)", 7, 90, 30)
        with col2:
            if st.button("Generate Prediction", type="primary"):
                predictions = predict_stock_price(df, days_ahead)
                
                # Create prediction dataframe
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                            periods=days_ahead)
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'][-90:],
                                        name="Historical", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions,
                                        name="Predicted", line=dict(color='red', dash='dash')))
                
                fig.update_layout(title=f"{ticker} Price Prediction",
                                 xaxis_title="Date", yaxis_title="Price ($)",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                st.success(f"**Predicted price in {days_ahead} days: ${predictions[-1]:.2f}**")
                
                pred_change = ((predictions[-1] - current_price) / current_price) * 100
                if pred_change > 0:
                    st.info(f"📈 Expected increase: {pred_change:.2f}%")
                else:
                    st.warning(f"📉 Expected decrease: {pred_change:.2f}%")
        
        st.markdown("---")
        st.info("⚠️ **Disclaimer**: This prediction is based on historical data and machine learning. It should not be used as financial advice.")
    
    # Tab 3: Technical Analysis
    with tab3:
        st.subheader("📈 Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            st.markdown("**Relative Strength Index (RSI)**")
            rsi_value = df['RSI'].iloc[-1]
            
            if rsi_value > 70:
                st.error(f"RSI: {rsi_value:.2f} - Overbought ⚠️")
            elif rsi_value < 30:
                st.success(f"RSI: {rsi_value:.2f} - Oversold 💰")
            else:
                st.info(f"RSI: {rsi_value:.2f} - Neutral")
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD
            st.markdown("**MACD (Moving Average Convergence Divergence)**")
            macd_value = df['MACD'].iloc[-1]
            signal_value = df['Signal'].iloc[-1]
            
            if macd_value > signal_value:
                st.success("MACD Signal: Bullish 📈")
            else:
                st.warning("MACD Signal: Bearish 📉")
            
            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal"))
            fig_macd.update_layout(height=300)
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands
        st.markdown("**Bollinger Bands**")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="Upper Band",
                                    line=dict(dash='dash')))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="Lower Band",
                                    line=dict(dash='dash')))
        fig_bb.update_layout(height=400)
        st.plotly_chart(fig_bb, use_container_width=True)
    
    # Tab 4: AI Chatbot
    with tab4:
        st.subheader("💬 AI Stock Assistant")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
        
        # User input
        user_input = st.chat_input("Ask me anything about stocks...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate response
            response = ai_chatbot_response(user_input, df)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            st.rerun()
    
    # Tab 5: News & Sentiment
    with tab5:
        st.subheader("📰 Latest News & Sentiment Analysis")
        
        news = get_stock_news(ticker)
        
        if news:
            overall_sentiment = 0
            for article in news:
                with st.expander(f"📄 {article['title']}"):
                    st.write(article.get('publisher', 'Unknown'))
                    
                    # Sentiment analysis
                    sentiment, score = analyze_sentiment(article['title'])
                    overall_sentiment += score
                    
                    if sentiment == "Positive":
                        st.success(f"Sentiment: {sentiment} ({score:.2f})")
                    elif sentiment == "Negative":
                        st.error(f"Sentiment: {sentiment} ({score:.2f})")
                    else:
                        st.info(f"Sentiment: {sentiment} ({score:.2f})")
                    
                    if 'link' in article:
                        st.markdown(f"[Read more]({article['link']})")
            
            # Overall sentiment
            avg_sentiment = overall_sentiment / len(news)
            st.markdown("---")
            st.subheader("Overall News Sentiment")
            
            if avg_sentiment > 0.1:
                st.success(f"✅ Positive ({avg_sentiment:.2f})")
            elif avg_sentiment < -0.1:
                st.error(f"⚠️ Negative ({avg_sentiment:.2f})")
            else:
                st.info(f"➖ Neutral ({avg_sentiment:.2f})")
        else:
            st.info("No recent news available for this stock.")

else:
    st.error(f"Unable to fetch data for ticker: {ticker}. Please check the ticker symbol.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ for College Project | Data provided by Yahoo Finance</p>
        <p><small>⚠️ This is for educational purposes only. Not financial advice.</small></p>
    </div>
""", unsafe_allow_html=True)
