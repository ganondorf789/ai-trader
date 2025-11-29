Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's VolatilitySurgeBreakout Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

class VolatilitySurgeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    vwap_period = 14
    volume_median_period = 2880  # 30 days in 15m intervals
    
    def init(self):
        # 🌙 Calculate core indicators
        self.typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        
        # VWAP Calculation (using pandas_ta)
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low, 
                          close=self.data.Close, 
                          volume=self.data.Volume, 
                          length=self.vwap_period,
                          name='VWAP')
        
        # Standard Deviation of Typical Price
        self.std_dev = self.I(talib.STDDEV, self.typical_price, timeperiod=self.vwap_period, name='STDDEV')
        
        # 🌙 Create VWAP Bands
        self.upper_2x = self.I(lambda: self.vwap + 2*self.std_dev, name='UPPER_2X')
        self.lower_2x = self.I(lambda: self.vwap - 2*self.std_dev, name='LOWER_2X')
        self.upper_1x = self.I(lambda: self.vwap + 1*self.std_dev, name='UPPER_1X')
        self.lower_1x = self.I(lambda: self.vwap - 1*self.std_dev, name='LOWER_1X')
        self.upper_0_5x = self.I(lambda: self.vwap + 0.5*self.std_dev, name='UPPER_0.5X')
        self.lower_0_5x = self.I(lambda: self.vwap - 0.5*self.std_dev, name='LOWER_0.5X')
        
        # Volume Median (30-day lookback)
        self.volume_median = self.I(talib.MEDIAN, self.data.Volume, timeperiod=self.volume_median_period, name='VOL_MEDIAN')
        
        print("🌙✨ Moon Dev's Volatility Surge Strategy initialized! 🚀")
        print("🌌 Indicators ready for cosmic trading opportunities ✨")

    def next(self):
        current_idx = len(self.data)-1
        if current_idx < max(self.vwap_period, self.volume_median_period) + 2:
            return
        
        # 🌙 Get previous bar values
        prev_close = self.data.Close[-2]
        prev_volume = self.data.Volume[-2]
        prev_upper_2x = self.upper_2x[-2]
        prev_lower_2x = self.lower_2x[-2]
        prev_vol_median = self.volume_median[-2]
        
        # Current trading price
        price = self.data.Open[-1]
        
        # 🌙 Risk Management Calculations
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        
        # 🚀 Long Entry Logic
        if not self.position.is_long and prev_close > prev_upper_2x and prev_volume > prev_vol_median:
            sl_price = self.lower_2x[-1]  # Opposite 2x band
            risk_per_share = price - sl_price
            
            if risk_per_share > 0:
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, tag="Moon Surge Long")
                    print(f"🌙🚀 COSMIC LONG SIGNAL! Entry: {price:.2f} | Size: {position_size} | SL: {sl_price:.2f} ✨")
                    print(f"🌕 Price breached upper band with strong volume - riding the lunar wave!")
        
        # 🌑 Short Entry Logic
        elif not self.position.is_short and prev_close < prev_lower_2x and prev_volume > prev_vol_median:
            sl_price = self.upper_2x[-1]  # Opposite 2x band
            risk_per_share = sl_price - price
            
            if risk_per_share > 0:
                position_size = int(round(risk_amount / risk_per_share))
                if position_size >