```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

# Data preprocessing
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class OscillatorDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    tp_mult = 1.03   # 3% take profit
    sl_mult = 0.97   # 3% stop loss
    
    def init(self):
        # Ichimoku components
        self.chikou_span = self.I(lambda h,l,c: talib.ICHIMOKU(h,l,c,9,26,52)[4],
                                  self.data.High, self.data.Low, self.data.Close,
                                  name='Chikou_Span')
        
        # MACD Histogram
        self.macd_hist = self.I(lambda c: talib.MACD(c,12,26,9)[2],
                                self.data.Close, name='MACD_Hist')
        
        # Stochastic %K
        self.stoch_k = self.I(lambda h,l,c: talib.STOCH(h,l,c,14,3,3)[0],
                              self.data.High, self.data.Low, self.data.Close,
                              name='Stoch_%K')

        # Swing calculations
        self.price_swing_high = self.I(talib.MAX, self.data.High, 20, name='Price_Swing_High')
        self.price_swing_low = self.I(talib.MIN, self.data.Low, 20, name='Price_Swing_Low')
        self.chikou_swing_high = self.I(talib.MAX, self.chikou_span, 20, name='Chikou_Swing_High')
        self.chikou_swing_low = self.I(talib.MIN, self.chikou_span, 20, name='Chikou_Swing_Low')
        self.macd_swing_high = self.I(talib.MAX, self.macd_hist, 20, name='MACD_Swing_High')
        self.macd_swing_low = self.I(talib.MIN, self.macd_hist, 20, name='MACD_Swing_Low')
        self.stoch_swing_high = self.I(talib.MAX, self.stoch_k, 20, name='Stoch_Swing_High')
        self.stoch_swing_low = self.I(talib.MIN, self.stoch_k, 20, name='Stoch_Swing_Low')

    def _bullish_divergence(self):
        price_low = self.price_swing_low[-1] < self.price_swing_low[-2]
        chikou_low = self.chikou_swing_low[-1] > self.chikou_swing_low[-2]
        macd_low = self.macd_swing_low[-1] > self.macd_swing_low[-2]
        stoch_low = self.stoch_swing_low[-1] > self.stoch_swing_low[-2]
        return all([price_low, chikou_low, macd_low, stoch_low])

    def _bearish_divergence(self):
        price_high = self.price_swing_high[-1] > self.price_swing_high[-2]
        chikou_high = self.chikou_swing_high[-1] < self.chikou_swing_high[-2]
        macd_high = self.macd_swing_high[-1] < self.macd_swing_high[-2]
        stoch_high = self.stoch_swing_high[-1] < self.stoch_swing_high[-2]
        return all([price_high, chikou_high, macd_high, stoch_high])

    def _long_confirmation(self):
        return (