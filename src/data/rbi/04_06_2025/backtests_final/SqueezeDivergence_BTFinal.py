I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's SqueezeDivergence Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import numpy as np
import talib
import pytz
from datetime import time

class SqueezeDivergence(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    bb_period = 20
    kc_period = 20
    kc_mult = 1.5
    swing_period = 20
    
    def init(self):
        # 🌙 Clean and prepare data
        self.data.df.index = pd.to_datetime(self.data.df.index)
        if self.data.df.index.tz is None:
            self.data.df.index = self.data.df.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        # 🌟 Core Indicators
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, self.bb_period, 2, 2
        )
        
        # Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, self.kc_period)
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.kc_period)
        self.kc_upper = self.I(lambda: self.ema + self.kc_mult * self.atr_kc)
        self.kc_lower = self.I(lambda: self.ema - self.kc_mult * self.atr_kc)
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Liquidation Zones
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Volume indicator
        self.volume_ema = self.I(talib.EMA, self.data.Volume, 20)
        
        # 🌙 Moon Dev Debug: Indicators initialized successfully!
        print("🌙✨ Moon Dev Indicators Activated! All systems go! ✨🌙")
        
    def next(self):
        # 🌙 Skip if not enough data
        if len(self.data) < 50:
            return
        
        # 🌟 Current Values
        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        current_time = self.data.index[-1].time()
        
        # 🚀 Entry Conditions
        # 1. Bollinger Squeeze
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]
        bb_widths = [(self.bb_upper[i] - self.bb_lower[i]) / self.bb_middle[i] for i in range(-20, 0)]
        squeeze = bb_width < np.percentile(bb_widths, 20)
        
        # 2. Volume Divergence
        vol_div = (high > self.data.High[-2]) and (self.volume_ema[-1] < self.volume_ema[-2])
        
        # 3. Liquidity Cluster
        liq_cluster = any(abs(close - lvl)/close < 0.005 
                         for lvl in [self.swing_high[-1], self.swing_low[-1]])
        
        # 4. Keltner Break
        trend_up = close > self.ema[-1]
        kc_break_short = close < self.kc_lower[-1] and trend_up
        kc_break_long = close > self.kc_upper[-1] and not trend_up
        
        # 5. Time Filter
        session_start = time(9, 30)
        session_end = time(12, 30)
        in_session = session_start <= current_time <= session_end
        
        # 🌙 Position Sizing
        atr_value = self.atr[-1]
        risk_amount = self.equity * self.risk_per_trade
        position_size = round(risk_amount / atr_value)  # Round to whole units
        
        # 🚀 Long