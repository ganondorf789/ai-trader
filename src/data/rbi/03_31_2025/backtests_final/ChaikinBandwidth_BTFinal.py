I'll fix the incomplete code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed improvements:

```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy

class ChaikinBandwidth(Strategy):
    def init(self):
        # Moon Dev Initialization Phase 🌙✨
        print("🌙 Moon Dev Strategy Initializing...")
        
        # Data preprocessing
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Indicator calculations with self.I()
        # Chaikin Money Flow (21-period)
        self.cmf = self.I(pta.cmf, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 21, name='CMF')
        
        # Bollinger Bands (20,2) and Bandwidth
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[0], self.data.Close, name='BB_Upper')
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[2], self.data.Close, name='BB_Lower')
        self.bb_mid = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[1], self.data.Close, name='BB_Mid')
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_mid, name='Bandwidth')
        
        # Keltner Channel (20,2xATR)
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR14')
        self.keltner_upper = self.I(lambda e, a: e + 2*a, self.ema20, self.atr14, name='Keltner_Upper')
        self.keltner_lower = self.I(lambda e, a: e - 2*a, self.ema20, self.atr14, name='Keltner_Lower')
        
        # Swing detection
        self.price_lows = self.I(talib.MIN, self.data.Low, 5, name='Price_Lows')
        self.cmf_lows = self.I(talib.MIN, self.cmf, 5, name='CMF_Lows')
        self.price_highs = self.I(talib.MAX, self.data.High, 5, name='Price_Highs')
        self.cmf_highs = self.I(talib.MAX, self.cmf, 5, name='CMF_Highs')
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        print("🌌 Indicators successfully initialized with Moon Dev precision!")

    def next(self):
        # Moon Dev risk management constants 🌙
        RISK_PCT = 0.01  # 1% risk per trade
        REWARD_RATIO = 2
        
        if len(self.data) < 100:  # Wait for sufficient data
            return
        
        # Current price and position status
        price = self.data.Close[-1]
        position_size = self.equity * RISK_PCT / (self.atr[-1] * REWARD_RATIO)
        position_size = round(position_size)  # Round to whole units
        
        # 1. Divergence detection
        bullish_div = (self.price_lows[-1] < self.price_lows[-2] and 
                      self.cmf_lows[-1] > self.cmf_lows[-2])
        bearish_div = (self.price_highs[-1] > self.price_highs[-2] and 
                      self.cmf_highs[-1] < self.cmf_highs[-2])
        
        # 2. Bandwidth contraction (lowest 20% of last 100 periods)
        bandwidth_window = self.bandwidth[-100:-1]
        if len(bandwidth_window) > 0:
            contraction = self.bandwidth[-1] < np.percentile(bandwidth_window, 20)
        else: