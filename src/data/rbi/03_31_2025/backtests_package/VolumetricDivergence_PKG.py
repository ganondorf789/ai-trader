Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumetricDivergence(Strategy):
    risk_pct = 0.01
    atr_period = 14
    vol_ma_period = 20
    divergence_window = 5
    fib_levels = [0.382, 0.5, 0.618]
    
    def init(self):
        # MACD components
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close, 
                                                            fastperiod=12, slowperiod=26, signalperiod=9,
                                                            name='MACD HIST 🌈')
        
        # Volume indicators
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period, name='VOL MA 📊')
        
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR 🌪️')
        self.atr_ma = self.I(talib.SMA, self.atr, 20, name='ATR MA 📉')
        
        # Price swings
        self.price_highs = self.I(talib.MAX, self.data.High, self.divergence_window, name='SWING HIGH 🔺')
        self.price_lows = self.I(talib.MIN, self.data.Low, self.divergence_window, name='SWING LOW 🔻')
        
        # MACD swings
        self.macd_highs = self.I(talib.MAX, self.macd_hist, self.divergence_window, name='MACD HIGH 🌟')
        self.macd_lows = self.I(talib.MIN, self.macd_hist, self.divergence_window, name='MACD LOW 🌠')
        
        # Divergence threshold
        self.macd_std = self.I(talib.STDDEV, self.macd_hist, 20, 1, name='MACD STD 📏')
        self.macd_mean = self.I(talib.SMA, self.macd_hist, 20, name='MACD MEAN 📐')

    def next(self):
        if self.position or len(self.data) < 20:
            return

        # Volatility filter
        if self.atr[-1] <= self.atr_ma[-1]:
            print("🌙 Moon Dev Alert: Volatility too low - skipping trade")
            return

        # Volume surge check
        if self.data.Volume[-1] < 1.5 * self.vol_ma[-1]:
            print("🌙 Moon Dev Alert: Volume insufficient - skipping trade")
            return

        # Bullish divergence detection
        bull_div = (self.price_lows[-1] < self.price_lows[-2] and 
                   self.macd_lows[-1] > self.macd_lows[-2] and
                   self.macd_hist[-1] > (self.macd_mean[-1] + 2*self.macd_std[-1]))

        # Bearish divergence detection
        bear_div = (self.price_highs[-1] > self.price_highs[-2] and 
                   self.macd_highs[-1] < self.macd_highs[-2] and
                   self.macd_hist[-1] < (self.macd_mean[-1] - 2*self.macd_std[-1]))

        if bull_div:
            print("✨ Moon Dev Signal: Bullish divergence detected - entering long")
            self._enter_long()
        elif bear_div:
            print("✨ Moon Dev Signal: Bearish divergence detected - entering short")
            self._enter_short()

    def _enter_long(self):
        swing_l