I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class CrossfireBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=5, name='RSI')
        self.current_day = None
        self.stop_loss = None

    def next(self):
        current_idx = len(self.data) - 1
        if current_idx < 200:  # Ensure enough data for indicators
            return

        # Check for new day for trailing stops
        current_date = self.data.index[-1].date()
        if self.current_day != current_date and self.current_day is not None:
            self.trail_stop()
        self.current_day = current_date

        # Entry logic
        if not self.position:
            self.check_entries()
        else:
            self.check_exits()

    def trail_stop(self):
        if self.position.is_long and len(self.data.Close) >= 2:
            prev_close = self.data.Close[-2]
            prev_atr = self.atr[-2]
            new_stop = prev_close - 2 * prev_atr
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                print(f"🌙✨ Trailing SL Updated: {self.stop_loss:.2f} | Moon Phase: Waxing")
        elif self.position.is_short and len(self.data.Close) >= 2:
            prev_close = self.data.Close[-2]
            prev_atr = self.atr[-2]
            new_stop = prev_close + 2 * prev_atr
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                print(f"🌙✨ Trailing SL Updated: {self.stop_loss:.2f} | Moon Phase: Waning")

    def check_entries(self):
        # Replaced crossover with direct array comparison
        ema50_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
        ema50_cross_under = (self.ema50[-2] > self.ema200[-2]) and (self.ema50[-1] < self.ema200[-1])
        adx_condition = self.adx[-1] > 25

        # Long entry conditions
        if ema50_cross and adx_condition and self.data.Close[-1] > self.data.Close[-2]:
            self.enter_long()

        # Short entry conditions
        elif ema50_cross_under and adx_condition and self.data.Close[-1] < self.data.Close[-2]:
            self.enter_short()

    def enter_long(self):
        risk_amount = self.equity * self.risk_per_trade
        atr_value = self.atr[-1]
        stop_loss_price = self.data.Close[-1] - 2 * atr_value
        risk_per_share = self.data.Close[-1] - stop_loss_price
        
        if risk_per_share <= 0:
            print("🌙⚠️ Risk per share invalid for LONG - Moon alignment unfavorable")
            return
            
        position_size = int(round(risk_amount / risk_per_share))
        if position_size > 0:
            self.buy(size=position_size)
            self.stop_loss = stop_loss_price
            print(f"🌙🚀✨ LONG ENTRY | Price: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {self.stop_loss:.2f} | Moon Power: Strong")

    def enter_short(self):
        risk_amount = self.equity * self.risk_per_trade
        atr_value = self.atr[-1]
        stop_loss_price = self.data