I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

class BandwidthSurge(Strategy):
    risk_percent = 1  # 1% risk per trade
    
    def init(self):
        # Moon Dev Indicator Setup 🌙
        print("🌙 Initializing Moon Dev BandwidthSurge Strategy...")
        print("✨ Loading celestial indicators from TA-Lib constellation...")
        
        # Convert to numpy arrays for TA-Lib
        close = self.data.Close.values.astype('float')
        high = self.data.High.values.astype('float')
        low = self.data.Low.values.astype('float')
        volume = self.data.Volume.values.astype('float')
        
        # 1. Bollinger Bands (20,2)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda: upper, name='Upper BB')
        self.middle_band = self.I(lambda: middle, name='Middle BB')
        self.lower_band = self.I(lambda: lower, name='Lower BB')
        
        # Calculate bandwidth
        bandwidth = (upper - lower) / middle
        self.bandwidth = self.I(lambda: bandwidth, name='Bandwidth')
        
        # 2. Volume SMA(20)
        self.volume_sma = self.I(lambda: talib.SMA(volume, timeperiod=20), name='Volume SMA')
        print("🌙 Volume SMA(20) initialized - tracking lunar tidal patterns...")
        
        # 3. RSI(14)
        self.rsi = self.I(lambda: talib.RSI(close, timeperiod=14), name='RSI')
        print("🌙 RSI(14) activated - measuring cosmic momentum...")
        
        # 4. ATR(14) for risk management
        self.atr = self.I(lambda: talib.ATR(high, low, close, timeperiod=14), name='ATR')
        print("🌙 ATR(14) online - calculating celestial risk boundaries...")
        
        # Swing indicators using TA-Lib
        self.bandwidth_10_low = self.I(lambda: talib.MIN(bandwidth, timeperiod=10), name='Bandwidth 10L')
        self.bandwidth_5_high = self.I(lambda: talib.MAX(bandwidth, timeperiod=5), name='Bandwidth 5H')
        print("🌙 Bandwidth oscillators calibrated - ready for launch sequence...")
        
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
                    print("🌙⚠️ ATR anomaly detected - aborting launch sequence!")
                    return  # Safety check
                
                # Risk management calculations
                risk_amount = self.equity * self.risk_percent / 100
                position_size = risk_amount / (atr_value * 1)  # 1x ATR stop
                position_size = int(round(position_size))  # Ensure whole number of units
                
                if position_size > 0:
                    stop_price = price - atr_value
                    take