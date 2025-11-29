```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import numpy as np

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeDivergence(Strategy):
    risk_pct = 0.01
    bb_period = 20
    atr_period = 14
    lookback = 100
    swing_window = 5
    
    def init(self):
        # Bollinger Bands
        close = self.data.Close
        upper, middle, lower = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: upper, name='BB_Upper')
        self.bb_middle = self.I(lambda: middle, name='BB_Middle')
        self.bb_lower = self.I(lambda: lower, name='BB_Lower')
        
        # Bollinger Bandwidth Percentile
        bb_width = (upper - lower) / middle
        self.bb_width_pct = self.I(lambda: bb_width.rolling(self.lookback).rank(pct=True)*100, name='BB_Width_Pct')
        
        # VWAP
        vwap = ta.vwap(high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume)
        self.vwap = self.I(lambda: vwap, name='VWAP')
        
        # ATR and Percentile
        atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr = self.I(lambda: atr, name='ATR')
        self.atr_pct = self.I(lambda: atr.rolling(self.lookback).rank(pct=True)*100, name='ATR_Pct')
        
        # Swing Highs/Lows
        self.price_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window, name='Price_High')
        self.vwap_high = self.I(talib.MAX, self.vwap, timeperiod=self.swing_window, name='VWAP_High')
        self.price_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window, name='Price_Low')
        self.vwap_low = self.I(talib.MIN, self.vwap, timeperiod=self.swing_window, name='VWAP_Low')
        
        # Volume-Weighted RSI
        rsi = talib.RSI(close, timeperiod=14)
        vw_rsi = rsi * self.data.Volume
        self.vw_rsi = self.I(lambda: talib.SMA(vw_rsi, timeperiod=5), name='VW_RSI')
        
    def next(self):
        if len(self.data) < self.lookback:
            return
        
        i = len(self.data)-1
        
        # Entry Logic
        if not self.position:
            # Bollinger Squeeze
            if self.bb_width_pct[i] < 20:
                # Divergence Check
                ph, vh = self.price_high[i], self.vwap_high[i]
                ph_prev, vh_prev = self.price_high[i-1], self.vwap_high[i-1]
                bear_div = ph > ph_prev and vh < vh_prev
                
                pl, vl = self.price_low[i], self.vwap_low[i]
                pl_prev, vl_prev = self.price_low[i-1], self.vwap_low[i-1]
                bull_div = pl < pl_prev and vl > vl_prev
                
                # Volatility Filter
                if self.atr_pct[i] > 80:
                    price = self.data.Close[i]
                    vwap = self.vwap[i]
                    
                    if bull_div and price > vwap:
                        sl = price - 1.5 * self.atr[i]
                        tp = price + 3 * self.atr