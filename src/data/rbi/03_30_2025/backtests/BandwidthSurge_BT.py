Here's the complete backtesting.py implementation for the BandwidthSurge strategy:

```python
# 🌙 Moon Dev's BandwidthSurge Backtest Implementation 🚀
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import pandas_ta as pta
import numpy as np

class BandwidthSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 1.5  # ATR multiplier for volatility adjustment

    def init(self):
        # 🌟 Indicator Calculation Section 🌟
        # Bollinger Bands Components
        self.bb_middle = self.I(talib.SMA, self.data.Close, 20, name='BB_MIDDLE')
        self.bb_std = self.I(talib.STDDEV, self.data.Close, 20, name='BB_STD')
        self.bb_upper = self.I(lambda c, m, s: m + 2*s, 
                              self.data.Close, self.bb_middle, self.bb_std,
                              name='BB_UPPER')
        self.bb_lower = self.I(lambda c, m, s: m - 2*s,
                              self.data.Close, self.bb_middle, self.bb_std,
                              name='BB_LOWER')
        
        # Bollinger Band Width Calculation
        self.bb_width = self.I(lambda u, l, m: (u - l)/m,
                              self.bb_upper, self.bb_lower, self.bb_middle,
                              name='BB_WIDTH')
        self.bb_width_low = self.I(talib.MIN, self.bb_width, 20,
                                  name='BB_WIDTH_LOW')

        # Klinger Volume Oscillator
        def calculate_kvo(h, l, c, v):
            kvo = pta.kvo(h, l, c, v, fast=34, slow=55)
            return kvo.iloc[:, 0]  # Returns KVO line
        
        self.kvo_line = self.I(calculate_kvo,
                              self.data.High, self.data.Low,
                              self.data.Close, self.data.Volume,
                              name='KVO_LINE')

        # Parabolic SAR
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low,
                         acceleration=0.02, maximum=0.2,
                         name='SAR')

        # ATR for volatility measurement
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                         self.data.Close, 14, name='ATR')

        print("🌙 BandwidthSurge Strategy Initialized with Moon Dev Magic! 🚀")

    def next(self):
        # 🌌 Moon Dev Debug Prints 🌌
        current_close = self.data.Close[-1]
        print(f"\n🌙 New Bar | Close: {current_close:.2f} | Equity: {self.equity:.2f} ✨")

        # Ensure sufficient data history
        if len(self.data.Close) < 20:
            return

        # 🚨 Entry Conditions 🚨
        bb_contraction = self.bb_width[-1] <= self.bb_width_low[-1]
        price_above_upper = current_close > self.bb_upper[-1]
        price_below_lower = current_close < self.bb_lower[-1]

        # Klinger Divergence Conditions
        kvo_rising = self.kvo_line[-2] < self.kvo_line[-1]
        kvo_falling = self.kvo_line[-2] > self.kvo_line[-1]
        price_falling = self.data.Close[-2] > current_close
        price_rising = self.data.Close[-2] < current_close

        # 💹 Long Entry Logic
        if not self.position.is_long and bb_contraction:
            if kvo_rising and price_falling and price_above_upper:
                risk_amount = self.equity * self.risk_percent
                sl_price = self.bb_lower[-1]
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price,
                                tag="MoonDev Long Entry 🌕")
                        print(f"🚀 LONG ENTRY | Size: {position_size}")
                        print(f"   Entry: {current_close:.2f} | SL: {sl_price:.2f}")
                        print(f"   Risk/Reward: 1