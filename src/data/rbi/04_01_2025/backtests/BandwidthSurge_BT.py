```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from talib import BBANDS, RSI, ATR, MIN, MAX, SMA

class BandwidthSurge(Strategy):
    risk_percent = 1  # 1% risk per trade
    
    def init(self):
        # Moon Dev Indicator Setup 🌙
        print("🌙 Initializing Moon Dev BandwidthSurge Strategy...")
        
        # Convert to numpy arrays for TA-Lib
        close = self.data.Close.values.astype('float')
        high = self.data.High.values.astype('float')
        low = self.data.Low.values.astype('float')
        volume = self.data.Volume.values.astype('float')
        
        # 1. Bollinger Bands (20,2)
        upper, middle, lower = BBANDS(close, 20, 2, 2)
        self.upper_band = self.I(lambda: upper, name='Upper BB')
        self.middle_band = self.I(lambda: middle, name='Middle BB')
        self.lower_band = self.I(lambda: lower, name='Lower BB')
        
        # Calculate bandwidth
        bandwidth = (upper - lower) / middle
        self.bandwidth = self.I(lambda: bandwidth, name='Bandwidth')
        
        # 2. Volume SMA(20)
        self.volume_sma = self.I(lambda: SMA(volume, 20), name='Volume SMA')
        
        # 3. RSI(14)
        self.rsi = self.I(lambda: RSI(close, 14), name='RSI')
        
        # 4. ATR(14) for risk management
        self.atr = self.I(lambda: ATR(high, low, close, 14), name='ATR')
        
        # Swing indicators using TA-Lib
        self.bandwidth_10_low = self.I(lambda: MIN(bandwidth, 10), name='Bandwidth 10L')
        self.bandwidth_5_high = self.I(lambda: MAX(bandwidth, 5), name='Bandwidth 5H')
        
    def next(self):
        # Moon Dev Strategy Logic 🌙✨
        if len(self.data) < 20:  # Wait for indicators to warm up
            return
            
        # Current values
        price = self.data.Close[-1]
        current_bandwidth = self.bandwidth[-1]
        current_volume = self.data.Volume[-1]
        current_volume_sma = self.volume_sma[-1]
        current_rsi = self.rsi[-1]
        previous_rsi = self.rsi[-2] if len(self.rsi) > 1 else 0
        
        # Entry Conditions 🌙🚀
        entry_condition_1 = current_bandwidth <= self.bandwidth_10_low[-1]
        entry_condition_2 = current_volume >= 1.5 * current_volume_sma
        entry_condition_3 = (current_rsi > 30) and (previous_rsi <= 30)
        
        # Exit Conditions 🌙💫
        exit_condition_1 = current_bandwidth >= self.bandwidth_5_high[-1]
        exit_condition_2 = current_rsi > 70
        
        # Moon Dev Position Management 🌙📈
        if not self.position:
            if all([entry_condition_1, entry_condition_2, entry_condition_3]):
                atr_value = self.atr[-1]
                if atr_value <= 0:
                    return  # Safety check
                
                # Risk management calculations
                risk_amount = self.equity * self.risk_percent / 100
                position_size = risk_amount / (atr_value * 1)  # 1x ATR stop
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=price - atr_value,
                            tp=price + (2 * atr_value))  # 1:2 RR
                    print(f"🌙🚀 MOON DEV LONG! Size: {position_size} | Entry: {price:.2f} | SL: {price - atr_value:.2f}")
        else:
            # Check exit conditions
            if exit_condition_1 or exit_condition_2:
                self.position.close()
                print(f"🌙💫 CLOSING POSITION | BW High: {exit_condition_1} | RSI Exit: {exit_condition_2}")

# Data preparation 🌙📊
data = pd.read_csv('/Users/md/Dropbox/dev