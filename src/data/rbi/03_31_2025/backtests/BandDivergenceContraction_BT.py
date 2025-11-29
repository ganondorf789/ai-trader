```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and prepare data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class BandDivergenceContraction(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands
        self.middle_band = self.I(talib.SMA, self.data.Close, 20, name='SMA20')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, 20, name='STD20')
        self.upper_band = self.I(lambda: self.middle_band + 2*self.std_dev, name='Upper BB')
        self.lower_band = self.I(lambda: self.middle_band - 2*self.std_dev, name='Lower BB')
        
        # Williams %R
        self.willr = self.I(talib.WILLR, self.data.High, self.data.Low, self.data.Close, 14, name='Williams%R')
        
        # Divergence detection components
        self.price_lows = self.I(talib.MIN, self.data.Low, 5, name='Price Lows')
        self.willr_lows = self.I(talib.MIN, self.willr, 5, name='WillR Lows')
        self.price_highs = self.I(talib.MAX, self.data.High, 5, name='Price Highs')
        self.willr_highs = self.I(talib.MAX, self.willr, 5, name='WillR Highs')
        
        # Bandwidth calculations
        self.bandwidth = self.I(lambda: (self.upper_band - self.lower_band)/self.middle_band, name='Bandwidth')
        self.bandwidth_sma = self.I(talib.SMA, self.bandwidth, 10, name='Bandwidth SMA')
        
        # Trend filter
        self.sma50 = self.I(talib.SMA, self.data.Close, 50, name='SMA50')
        
        # Swing points for stop loss
        self.swing_low_20 = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')
        self.swing_high_20 = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        
    def next(self):
        if len(self.data.Close) < 50:  # Wait for indicators to warm up
            return
            
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Bullish divergence check
        bull_div = (self.price_lows[-1] < self.price_lows[-2] and 
                   self.willr_lows[-1] > self.willr_lows[-2] and 
                   self.willr[-1] < -80)
        
        # Bearish divergence check
        bear_div = (self.price_highs[-1] > self.price_highs[-2] and 
                   self.willr_highs[-1] < self.willr_highs[-2] and 
                   self.willr[-1] > -20)
        
        # Trend filter
        uptrend = current_close > self.sma50[-1]
        downtrend = current_close < self.sma50[-1]
        
        # Entry logic
        if not self.position:
            # Long entry
            if bull_div and current_low <= self.lower_band[-1] and uptrend:
                sl_price = min(self.swing_low_20[-1], self.lower_band[-1])
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk