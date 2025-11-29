Here's the fixed code with all backtesting.lib imports removed and replaced with proper Moon Dev compliant implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION CONTRARIAN STRATEGY 🚀
# 🌟 FULLY COMPLIANT WITH MOON DEV PACKAGE STANDARDS ✨

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class LiquidationContrarian(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌕
    max_concurrent_trades = 3
    daily_loss_limit = 0.05
    weekly_loss_limit = 0.10
    
    def init(self):
        # 🌙 Calculate indicators using TA-Lib with self.I() wrapper
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING LOW')
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        
        # 🌙 Initialize risk management trackers
        self.daily_loss = 0
        self.weekly_loss = 0
        self.last_day = None
        self.last_week = None

    def next(self):
        # 🌙 Reset daily/weekly counters at period boundaries
        current_day = self.data.index[-1].date()
        if self.last_day != current_day:
            self.daily_loss = 0
            self.last_day = current_day
            
        current_week = self.data.index[-1].isocalendar()[1]
        if self.last_week != current_week:
            self.weekly_loss = 0
            self.last_week = current_week

        # 🌙 Check loss limits
        if (self.daily_loss <= -self.daily_loss_limit or 
            self.weekly_loss <= -self.weekly_loss_limit):
            print("🌧️ MOON DEV ALERT: Loss limit reached! Trading halted.")
            return

        # 🌙 Check open positions count
        if len(self.trades) >= self.max_concurrent_trades:
            return

        # 🌙 Main strategy logic
        current_idx = len(self.data)-1
        
        # Bullish entry check
        if not self.position.is_long and self._bullish_setup(current_idx):
            self._enter_long(current_idx)
            
        # Bearish entry check
        if not self.position.is_short and self._bearish_setup(current_idx):
            self._enter_short(current_idx)

    def _bullish_setup(self, i):
        # 🌙 Liquidation level rejection check
        swing_low = self.swing_low[i-1]
        if (self.data.Low[i-1] > swing_low * 1.01 or
            self.data.Close[i] < swing_low * 1.03):
            return False

        # 🌙 Divergence check (3-period)
        price_lows = [self.data.Low[j] for j in [i-2, i-1, i]]
        obv_lows = [self.obv[j] for j in [i-2, i-1, i]]
        rsi_lows = [self.rsi[j] for j in [i-2, i-1, i]]

        return (price_lows[0] > price_lows[1] > price_lows[2] and
                obv_lows[0] < obv_lows[1] < obv_lows[2] and
                rsi_lows[0] < rsi_lows[1] < rsi_lows[2])

    def _bearish_setup(self, i):
        # 🌙 Liquidation level rejection check
        swing_high = self.swing_high[i-1]
        if (self.data.High[i-1] < swing_high * 0.99 or
            self.data.Close[i] > swing_high * 0.97):
            return False

        # 🌙 Divergence check (3-period)
        price_highs = [self.data.High[j] for j in [i-2, i-1, i]]
        obv_highs = [self.obv[j] for j in [i-2, i-1, i]]
        rsi_highs = [self.rsi[j] for j in [i-2, i-1, i]]

        return (price