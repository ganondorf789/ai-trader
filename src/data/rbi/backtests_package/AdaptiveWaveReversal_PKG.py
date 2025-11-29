Below is the fixed code with proper Moon Dev themed debug prints and without any usage of backtesting.lib:

```python
#!/usr/bin/env python3
"""
Moon Dev’s Backtest AI 🌙
Strategy: AdaptiveWaveReversal
--------------------------------------------------
Instructions:
• Cleans the CSV column names and drops unnamed columns.
• Uses TA‑lib for indicator calculations via self.I().
• Implements long/short entries based on a higher timeframe “trend” (approximated using VWAP),
  swing highs/lows (via talib.MAX/min), and pivot points for dynamic support/resistance.
• Uses risk management (1% risk per trade on a 1,000,000 account) and proper integer position sizing.
• Prints themed debug messages.
• Finally, prints full backtest stats.
--------------------------------------------------
NOTE: No charting is invoked. Only stats are printed.
"""

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest

# ***********************************************
# Helper functions using TA-lib / pandas for indicators
# ***********************************************

def calc_vwap(close, high, low, volume):
    # VWAP computation using typical price and cumulative sum.
    # Typical Price = (High+Low+Close)/3
    typical_price = (high + low + close) / 3.0
    cumulative_tp_volume = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    vwap = cumulative_tp_volume / cumulative_volume
    return vwap

def calc_pivot(high, low, close):
    # Standard pivot point: (High + Low + Close)/3
    return (high + low + close) / 3.0

# ***********************************************
# Strategy Class Definition
# ***********************************************

class AdaptiveWaveReversal(Strategy):
    # Set the default cash size to 1,000,000
    cash = 1_000_000

    def init(self):
        # *******************************
        # Compute Indicators via self.I wrapper
        # *******************************
        # Compute Swing High (rolling maximum) and Swing Low (rolling minimum) over a 20-bar window
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low  = self.I(talib.MIN,  self.data.Low,  timeperiod=20)

        # Compute VWAP using our custom function - note: VWAP is not available in TA‑lib so we use our function.
        self.vwap = self.I(calc_vwap, self.data.Close, self.data.High, self.data.Low, self.data.Volume)
        
        # Compute Pivot value using our custom pivot function (for an extra dynamic level)
        self.pivot = self.I(calc_pivot, self.data.High, self.data.Low, self.data.Close)

        # Debug print for initialization
        print("🌙🚀 [Moon Dev] Strategy Initialized with indicators!")

    def next(self):
        price = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        current_pivot = self.pivot[-1]
        # Using previous swing values for entry thresholds (note: index -2 because current bar might not have closed fully)
        prev_swing_high = self.swing_high[-2] if len(self.swing_high) > 1 else None
        prev_swing_low = self.swing_low[-2] if len(self.swing_low) > 1 else None

        # Determine trend direction using VWAP
        if price > current_vwap:
            trend = "uptrend"
        else:
            trend = "downtrend"
        print(f"🌙✨ [Moon Dev] Bar: {self.data.index[-1]} | Price: