Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints added:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Moon Dev's BandwidthMomentum Strategy 🌙✨
class BandwidthMomentum(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    bandwidth_expansion_factor = 1.2  # 20% expansion threshold
    
    def init(self):
        # Clean and prepare data columns
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # Calculate Bollinger Bands components
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=20, name='Middle BB')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=20, nbdev=1, name='StdDev')
        self.upper_band = self.I(lambda m, s: m + 2*s, self.middle_band, self.std_dev, name='Upper BB')
        self.lower_band = self.I(lambda m, s: m - 2*s, self.middle_band, self.std_dev, name='Lower BB')
        
        # Calculate Bollinger Bandwidth
        self.bandwidth = self.I(lambda u, l, m: ((u - l)/m)*100, 
                               self.upper_band, self.lower_band, self.middle_band,
                               name='Bandwidth')
        self.bandwidth_sma = self.I(talib.SMA, self.bandwidth, timeperiod=20, name='Bandwidth SMA')
        
        # Momentum indicators
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14, name='CMO(14)')
        self.signal_line = self.I(talib.SMA, self.cmo, timeperiod=9, name='Signal Line')
        
        # Swing calculations for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        
        print("🌙✨ Moon Dev indicators initialized! Ready for cosmic analysis!")

    def next(self):
        # Skip first 20 periods to allow indicator calculations
        if len(self.data) < 20:
            return
        
        # Current and previous values for crossover detection
        current_cmo = self.cmo[-1]
        prev_cmo = self.cmo[-2]
        current_signal = self.signal_line[-1]
        prev_signal = self.signal_line[-2]
        
        # Entry conditions
        if not self.position:
            # Check for contraction phase
            contraction = self.bandwidth[-1] < self.bandwidth_sma[-1]
            
            # Long entry: CMO crosses above signal line during contraction
            if (prev_cmo < prev_signal and current_cmo > current_signal) and contraction:
                entry_price = self.data.Open[-1]
                swing_low = self.swing_low[-1]
                stop_loss = swing_low * 0.995
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss, tag='Long Entry')
                        self.entry_bandwidth = self.bandwidth[-1]
                        print(f"🌕🚀 BULLISH ALERT! Long entry at {entry_price:.2f} "
                              f"Size: {position_size} contracts")

            # Short entry: CMO crosses below signal line during contraction
            elif (prev_cmo > prev_signal and current_cmo < current_signal) and contraction:
                entry_price = self.data.Open[-1]
                swing_high = self.swing_high[-1]
                stop_loss = swing_high * 1.005
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_loss, tag='Short Entry')
                        self.entry_bandwidth