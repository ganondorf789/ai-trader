Here's the fixed code with all backtesting.lib references removed and replaced with proper Moon Dev themed implementations:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

class VortexConfirmation(Strategy):
    vi_length = 14
    ema_period = 20
    atr_period = 14
    risk_percent = 1  # 1% risk per trade
    atr_threshold = 50  # Adjust based on asset volatility

    def init(self):
        # 🌙 Vortex Indicator Calculation
        self.vi_plus, self.vi_minus = self.I(ta.vortex,
                                             self.data.High,
                                             self.data.Low,
                                             self.data.Close,
                                             length=self.vi_length)
        
        # 📈 Volume Confirmation
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        
        # 🌗 Keltner Channel Components
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         timeperiod=self.atr_period)

    def next(self):
        if self.position:
            return  # 🌌 Only one position at a time

        # 🌠 Current Indicator Values
        vi_cross = (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1])
        vi_reverse = (self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1])
        
        obv_trend_up = (self.obv[-1] > self.obv[-2] > self.obv[-3])
        obv_trend_down = (self.obv[-1] < self.obv[-2] < self.obv[-3])
        
        current_close = self.data.Close[-1]
        atr_val = self.atr[-1]
        kc_upper = self.ema20[-1] + 2*atr_val
        kc_lower = self.ema20[-1] - 2*atr_val

        # 🚀 Long Entry Logic
        if vi_cross and obv_trend_up:
            risk_amount = self.equity * self.risk_percent / 100
            stop_distance = 2 * atr_val
            position_size = int(round(risk_amount / stop_distance))
            
            if position_size > 0:
                self.buy(size=position_size,
                        sl=current_close - stop_distance,
                        tag="🌙 LONG: Vortex+Cross & OBV Rising")
                print(f"🌙🚀 BUY Signal @ {current_close:.2f} | Size: {position_size} | SL: {current_close - stop_distance:.2f}")

        # 🌪️ Short Entry Logic
        elif vi_reverse and obv_trend_down:
            risk_amount = self.equity * self.risk_percent / 100
            stop_distance = 2 * atr_val
            position_size = int(round(risk_amount / stop_distance))
            
            if position_size > 0:
                self.sell(size=position_size,
                         sl=current_close + stop_distance,
                         tag="🌙 SHORT: Vortex-Cross & OBV Falling")
                print(f"🌙🌪️ SELL Signal @ {current_close:.2f} | Size: {position_size} | SL: {current_close + stop_distance:.2f}")

        # 💫 Exit Logic
        if self.position.is_long and (current_close < kc_lower) and (atr_val < self.atr_threshold):
            self.position.close()
            print(f"🌙🔻 EXIT LONG @ {current_close:.2f} | Below KC & Low Volatility")
            
        elif self.position.is_short and (current_close > kc_upper) and (atr_val < self.atr_threshold):
            self.position.close()
            print(f"🌙🔼 EXIT SHORT @ {current_close:.2f} | Above KC & Low Volatility")

# 🌙 Data Preparation Ritual
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns