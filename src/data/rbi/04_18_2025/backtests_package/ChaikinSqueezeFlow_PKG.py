Here's the complete fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev Backtest AI Implementation 🌙
import pandas as pd
import talib
import pandas_ta
from backtesting import Backtest, Strategy
import numpy as np

class ChaikinSqueezeFlow(Strategy):
    timeperiod_3mo = 8640  # 3 months in 15m intervals (90 days * 96 periods/day)
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        print("🌙 Initializing Moon Dev Strategy with Celestial Indicators...")
        
        # Chaikin Money Flow (21-day period)
        self.cmf = self.I(pandas_ta.cmf,
                          high=self.data.High, low=self.data.Low,
                          close=self.data.Close, volume=self.data.Volume,
                          length=21*96, name='CMF_21D')
        
        # 5-day SMA of CMF
        self.cmf_sma = self.I(talib.SMA, self.cmf,
                             timeperiod=5*96, name='CMF_SMA_5D')
        
        # Bollinger Bands & Bandwidth
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_UP', 'BB_MID', 'BB_LOW']
        )
        self.bb_width = self.I(lambda u, l, m: (u - l)/m,
                              self.bb_upper, self.bb_lower, self.bb_middle,
                              name='BB_WIDTH')
        self.bb_width_min = self.I(talib.MIN, self.bb_width,
                                  self.timeperiod_3mo, name='BB_WIDTH_MIN')
        
        # VWAP (20-period)
        self.vwap = self.I(pandas_ta.vwap,
                          high=self.data.High, low=self.data.Low,
                          close=self.data.Close, volume=self.data.Volume,
                          length=20, name='VWAP_20')
        
        # ATR & Filter
        self.atr = self.I(talib.ATR,
                         self.data.High, self.data.Low, self.data.Close,
                         14, name='ATR_14')
        self.atr_sma = self.I(talib.SMA, self.atr,
                             timeperiod=30*96, name='ATR_SMA_30D')
        
        # Swing Low Detection
        self.swing_low = self.I(talib.MIN, self.data.Low,
                               timeperiod=20, name='SWING_LOW_20')
        
        self.consecutive_below = 0
        print("✨ Celestial Indicators Activated! Ready for Launch 🚀")

    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Processing Bar {len(self.data)} | Price: {price:.2f}")
        
        if not self.position:
            # Check volatility filter
            if self.atr[-1] <= self.atr_sma[-1]:
                print("🌊 Volatility Too Low - Cosmic Winds Calm ⛵")
                return

            # Entry conditions - replaced crossover with manual check
            cmf_bullish = (self.cmf_sma[-2] <= 0 and self.cmf_sma[-1] > 0)
            squeeze_on = (self.bb_width[-1] <= self.bb_width_min[-1])

            if cmf_bullish and squeeze_on:
                print("🌟 Bullish CMF Cross Detected in Squeeze Zone!")
                entry_price = price
                atr_value = self.atr[-1]
                
                # Calculate stop levels
                swing_stop = self.swing_low[-1] - 0.01
                atr_stop = entry_price - 2 * atr_value
                stop_price = max(swing_stop, atr_stop)
                
                # Calculate position size
                risk_per_share = entry_price - stop_price
                if risk_per_share <= 0:
                    print("🚨 Risk Calculation Error - Black Hole Avoided 🌌")
                    return
                
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    take_profit = entry_price + 2 * risk_per_share
                    self.buy(size=position_size,
                            sl=stop_price,
                            tp=take_profit)
                    print(f"🚀 LONG LAUN