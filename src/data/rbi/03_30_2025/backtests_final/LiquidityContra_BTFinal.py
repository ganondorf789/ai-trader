Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquidityContra(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (percentage of equity)
    swing_period = 20
    bandwidth_period = 20
    contraction_lookback = 100
    expansion_factor = 1.5  # 50% increase from contraction low
    max_bars_held = 15  # Time-based exit after 15 bars
    
    def init(self):
        # Clean and map data columns already handled in data preparation
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        
        # Calculate Bollinger Bands and bandwidth
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=self.bandwidth_period)
        self.bandwidth = self.I(lambda x: (upper - lower) / middle, name='BANDWIDTH')
        
        # Contraction threshold (20th percentile approximation)
        self.max_bw = self.I(talib.MAX, self.bandwidth, timeperiod=self.contraction_lookback, name='MAX_BW')
        self.min_bw = self.I(talib.MIN, self.bandwidth, timeperiod=self.contraction_lookback, name='MIN_BW')
        self.contraction_threshold = self.I(lambda x: self.min_bw + 0.2*(self.max_bw - self.min_bw), name='CONTRACTION_THRESH')
        
        # Track trade state
        self.entry_price = None
        self.trigger_index = -np.inf
        self.contraction_min = np.inf
        
    def next(self):
        if self.position:
            # Check time-based exit
            if len(self) - self.trigger_index >= self.max_bars_held:
                self.position.close()
                print(f"🌙 MOON DEV TIMEOUT EXIT: Closing position after {self.max_bars_held} bars")
            return
        
        # Check for liquidity zone breach
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        prev_swing_high = self.swing_high[-2]
        prev_swing_low = self.swing_low[-2]
        
        # Long trigger (price breaks below swing low)
        if self.data.Low[-2] > self.swing_low[-2] and self.data.Low[-1] < self.swing_low[-1]:
            print(f"🚀 LIQUIDITY BREACH DETECTED! Swing Low Broken at {prev_swing_low}")
            if self._check_contraction():
                self.contraction_min = self.bandwidth[-self.swing_period:-1].min()
                self.trigger_index = len(self)
                print(f"✨ CONTRACTION CONFIRMED! Min Bandwidth: {self.contraction_min:.4f}")
        
        # Short trigger (price breaks above swing high)
        elif self.data.High[-2] < self.swing_high[-2] and self.data.High[-1] > self.swing_high[-1]:
            print(f"🚀 LIQUIDITY BREACH DETECTED! Swing High Broken at {prev_swing_high}")
            if self._check_contraction():
                self.contraction_min = self.bandwidth[-self.swing_period:-1].min()
                self.trigger_index = len(self)
                print(f"✨ CONTRACTION CONFIRMED! Min Bandwidth: {self.contraction_min:.4f}")
        
        # Check volatility expansion after trigger
        if len(self) - self.trigger_index == 3:  # Allow 3 bars for expansion
            if self.bandwidth[-1] >= self.contraction_min * self.expansion_factor:
                if current_low < prev_swing_low:
                    self._enter_long()
                elif current_high > prev_swing_high:
                    self._enter_short()
            self.trigger_index = -np.inf  # Reset trigger
            
    def _check_contraction(self):
        return all(bw < self.contraction_threshold[-i-1] 
                  for i, bw in enumerate(self.bandwidth[-self.swing_period:-1]))
    
    def _enter_long(self):
        stop_price