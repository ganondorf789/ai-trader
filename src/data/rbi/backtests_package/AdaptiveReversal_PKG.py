#!/usr/bin/env python3
"""
Moon Dev's Backtest AI 🌙 - Adaptive Reversal Strategy Backtest
This script loads BTC-USD-15m data from the specified CSV, cleans and maps columns,
defines an Adaptive Reversal trading strategy with TA-Lib indicators and proper risk 
management with Moon Dev themed debug prints, then runs a backtest using a custom backtest function.
"""

# 1. All necessary imports
import pandas as pd
import numpy as np
import talib
from pandas_ta import MIN, MAX

# Custom backtest function
def backtest(data, strategy):
    cash = 1_000_000
    equity = cash
    position_size = 0
    realized_pnl = 0

    for idx in range(len(data)):
        close = data.Close[idx]
        sma20 = strategy.sma20[idx]
        low20 = strategy.low20[idx]
        high20 = strategy.high20[idx]

        if not position_size:
            if close <= low20:
                stop_loss = close * strategy.sl_factor
                risk_per_unit = close - stop_loss
                if risk_per_unit <= 0:
                    print("🚀🌙 [Moon Dev Warning] Computed risk per unit is non-positive. Skipping order!")
                    continue

                raw_position_size = equity * strategy.risk_percent / risk_per_unit
                position_size = round(raw_position_size)
                if position_size <= 0:
                    print("🚀🌙 [Moon Dev Warning] Calculated position size is zero. Skipping order!")
                    continue

                print(f"🚀🌙 [Moon Dev Signal] BUY triggered at {close:.2f} with stop-loss {stop_loss:.2f} "
                      f"and position size {position_size} (risk per unit: {risk_per_unit:.2f}).")
                position_price = close

        if position_size:
            if close > sma20:
                realized_pnl = position_size * (close - position_price)
                print(f"🌙✨ [Moon Dev Signal] SELL triggered at {close:.2f} as price recovers above SMA20 "
                      f"({sma20:.2f}). Exiting position with realized_pnl: {realized_pnl:.2f}.")
                equity += realized_pnl
                position_size = 0

        equity += (close - position_price) * position_size

    return equity

# 2. Define the Adaptive Reversal Strategy Class
class AdaptiveReversal:
    """
    Adaptive Reversal Strategy:
    A meta-strategy to cultivate self-learning and adaptability. For demo purposes,
    we trade when price dips to its recent low (anticipating a reversal upward)
    and exit when price recovers above its 20-bar SMA.
    """
    
    # You can set risk percent here (1% of account risk per trade)
    risk_percent = 0.01  
    sl_factor = 0.995  # Stop loss level as a fraction of entry price (0.5% risk buffer)

    def __init__(self, data):
        self.data = data

    def initialize_indicators(self):
        # Always use self.I() wrapper for indicator calculations 🌙
        # 20-period Simple Moving Average on Close using TA-Lib
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        # 20-bar lowest low using TA-Lib MIN function
        self.low20 = self.I(MIN, self.data.Low, timeperiod=20)
        # 20-bar highest high (not used in this demo but computed for reference)
        self.high20 = self.I(MAX, self.data.