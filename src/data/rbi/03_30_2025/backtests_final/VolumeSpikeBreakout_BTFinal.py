I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev themed prints:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
import numpy as np

class VolumeSpikeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    brick_size_multiplier = 1.5
    atr_period = 14
    cmf_period = 20
    volume_filter_period = 20

    def init(self):
        # Calculate indicators
        self.vwma50 = self.I(ta.vwma, self.data.Close, self.data.Volume, 50, name='VWMA50')
        self.vwma200 = self.I(ta.vwma, self.data.Close, self.data.Volume, 200, name='VWMA200')
        self.cmf = self.I(ta.cmf, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.cmf_period, name='CMF')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_filter_period, name='Volume_SMA')
        self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # Calculate brick size once
        self.brick_size = self.brick_size_multiplier * np.nanmean(self.atr)
        print(f"🌙 Moon Dev Initialization Complete | Renko Brick Size: {self.brick_size:.2f} ✨")

        # Renko tracking variables
        self.renko_direction = 1  # 1=up, -1=down
        self.renko_high = self.data.High[0] + self.brick_size
        self.renko_low = self.data.High[0]
        self.last_reversal_level = None

    def next(self):
        # Update Renko structure
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        if self.renko_direction == 1:
            if current_high >= self.renko_high + self.brick_size:
                self.renko_high += self.brick_size
                self.renko_low = self.renko_high - self.brick_size
            elif current_low <= self.renko_low - self.brick_size:
                self.last_reversal_level = self.renko_low
                self.renko_high = self.renko_low
                self.renko_low -= self.brick_size
                self.renko_direction = -1
                print(f"🌘 Moon Phase Shift | Renko Reversal DOWN at {self.last_reversal_level:.2f}")
        else:
            if current_low <= self.renko_low - self.brick_size:
                self.renko_low -= self.brick_size
                self.renko_high = self.renko_low + self.brick_size
            elif current_high >= self.renko_high + self.brick_size:
                self.last_reversal_level = self.renko_high
                self.renko_low = self.renko_high
                self.renko_high += self.brick_size
                self.renko_direction = 1
                print(f"🌖 Moon Phase Shift | Renko Reversal UP at {self.last_reversal_level:.2f}")

        # Entry logic
        if not self.position:
            # Check volume filter
            volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
            if not volume_ok:
                return

            # Check crossover
            vwma_cross = (self.vwma50[-2] < self.vwma200[-2] and self.vwma50[-1] > self.vwma200[-1])
            cmf_ok = self.cmf[-1] > 0.25

            if vwma_cross and cmf_ok:
                if self.last_reversal_level is None:
                    print("⚠️ Lunar Warning | No Reversal Level Detected - Skipping Trade")
                    return

                entry_price = self.data.Open[-1]
                stop_loss = self.last_reversal_level
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share <= 0:
                    print("⚠️ Cosmic Anomaly | Negative Risk Detected - Aborting Trade")
                    return

                # Calculate