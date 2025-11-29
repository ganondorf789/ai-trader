I'll fix the backtesting.lib imports and ensure proper indicator usage according to Moon Dev standards. Here's the corrected code:

```python
# 🌙 Moon Dev Backtest AI Implementation ✨
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VoltaicDivergence(Strategy):
    def init(self):
        # 🌀 Calculate Indicators with Cosmic Precision
        # Bollinger Bands (20,2)
        self.upper, self.mid, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                matype=0, name='BBANDS')
        
        # 🌌 Bollinger Band Width Analysis
        self.bb_width = self.I(lambda u,l: u-l, self.upper, self.lower, name='BB_Width')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 20, name='BB_Width_MA20')
        
        # 🌠 VWAP with Celestial Alignment (Daily Anchor)
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, 
                          self.data.Volume, anchor='D', name='VWAP')
        
        # ⚡ ATR for Stellar Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                        14, name='ATR14')
        
        # 📈 Volume Oscillator Filter
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20, name='Vol_MA20')

    def next(self):
        # 🌑 Skip initial cosmic formation period
        if len(self.data) < 20 or len(self.vwap) < 2:
            return
            
        # 🚫 Filter low-volume black holes
        if self.data.Volume[-1] < 0.5 * self.vol_ma[-1]:
            return

        # 🌙 Current Galactic Positions
        price = self.data.Close[-1]
        vwap_now = self.vwap[-1]
        vwap_prev = self.vwap[-2]
        bb_width = self.bb_width[-1]
        bb_width_avg = self.bb_width_avg[-1]

        # 🌗 Check for supernova volatility expansion
        if bb_width <= bb_width_avg:
            return

        # 🪐 Position Management
        if self.position:
            if self.position.is_long:
                # 🛸 Long exit: VWAP breach or energy contraction
                if (self.data.Close[-1] < self.vwap[-1] and self.data.Close[-2] >= self.vwap[-2]) or (bb_width < bb_width_avg):
                    self.position.close()
                    print(f"🌙 Moon Exit: Closed LONG at {price:.2f} ✨")
            else:
                # 🌑 Short exit: VWAP breach or energy contraction
                if (self.data.Close[-1] > self.vwap[-1] and self.data.Close[-2] <= self.vwap[-2]) or (bb_width < bb_width_avg):
                    self.position.close()
                    print(f"🌙 Moon Exit: Closed SHORT at {price:.2f} ✨")
        else:
            # 🚀 Long Entry: Upper band breakout with VWAP ascension
            if price > self.upper[-1] and vwap_now > vwap_prev:
                atr_val = self.atr[-1]
                risk_amount = 0.02 * self.equity
                position_size = int(round(risk_amount / (1.5 * atr_val)))
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=price - 1.5*atr_val,
                            tag="Celestial Ascent")
                    print(f"🚀 Moon Launch: LONG {position_size} @ {price:.2f} ✨")

            # 🌒 Short Entry: Lower band breakdown with VWAP descent
            elif price < self.lower[-1] and vwap_now < vwap_prev:
                atr_val = self.atr[-1]
                risk_amount = 0.02 * self.equity
                position_size = int(round(risk_amount / (1.5 * atr_val)))
                if position_size > 0:
                    self.sell(size=position_size,
                             sl=price + 1.5*atr_val,
                             tag="Stellar Descent")
                    print(f"🌒 Moon Dive: SHORT {position_size} @ {price:.