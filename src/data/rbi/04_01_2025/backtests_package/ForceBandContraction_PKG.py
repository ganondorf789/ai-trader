Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class ForceBandContraction(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    
    def init(self):
        # Clean data columns and handle unnamed columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate Elder Force Index (14-period EMA)
        close_diff = self.I(talib.MOM, self.data.Close, timeperiod=1, name='Momentum')
        volume = self.data.Volume
        force_raw = close_diff * volume
        self.force_index = self.I(talib.EMA, force_raw, timeperiod=14, name='ForceIndex')
        
        # Force Index 20-period SMA and StdDev
        self.force_ma = self.I(talib.SMA, self.force_index, timeperiod=20, name='ForceMA')
        self.force_std = self.I(talib.STDDEV, self.force_index, timeperiod=20, name='ForceStd')
        
        # Bollinger Bands and Bandwidth
        upper, middle, lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, 
                                      nbdevup=2, nbdevdn=2, matype=0, 
                                      name=['UpperBB', 'MiddleBB', 'LowerBB'])
        bandwidth = (upper - lower) / middle * 100
        self.bandwidth = self.I(lambda x: x, bandwidth, name='Bandwidth')
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, timeperiod=20, name='BandwidthLow')
        
        # Volatility stop (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                          timeperiod=14, name='ATR')
        
        print("🌙 MOON DEV INIT COMPLETE ✨ Indicators ready for launch!")

    def next(self):
        current_idx = len(self.data)-1
        
        # Skip initial warmup period
        if current_idx < 20:
            return
            
        # Calculate thresholds dynamically
        upper_threshold = self.force_ma[current_idx] + 2*self.force_std[current_idx]
        lower_threshold = self.force_ma[current_idx] - 2*self.force_std[current_idx]
        
        # Get bandwidth conditions
        bandwidth_condition = (self.bandwidth[current_idx] <= 
                              self.bandwidth_low[current_idx])
        
        # Moon Dev themed debug messages
        if not self.position:
            # Long entry check (replaced crossover with manual check)
            if (self.force_index[current_idx-1] < upper_threshold and
                self.force_index[current_idx] > upper_threshold and
                bandwidth_condition):
                
                atr = self.atr[current_idx]
                if atr == 0:
                    return
                
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.atr_multiplier * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    sl = self.data.Close[current_idx] - self.atr_multiplier * atr
                    self.buy(size=position_size, sl=sl)
                    print(f"🌙 BULLISH BREAKOUT DETECTED 🚀 | LONG {position_size} units "
                          f"| Price: {self.data.Close[current_idx]:.2f} | "
                          f"SL: {sl:.2f}")

            # Short entry check (replaced crossover with manual check)
            elif (self.force_index[current_idx-1] > lower_threshold and
                  self.force_index[current_idx] < lower_threshold and
                  bandwidth_condition):
                  
                atr = self.atr[current_idx]
                if atr == 0:
                    return
                
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.atr_multiplier * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    sl = self.data.Close[current_idx] + self.atr_multiplier * atr