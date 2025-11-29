```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest
import os

# 🌙 MOON DEV DATA PREPARATION 🌙
def load_data(path):
    # Load and clean data
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Map columns to backtesting.py format
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

class LiquidationBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    swing_period = 20  # Liquidation cluster lookback
    volume_sma_period = 20
    
    def init(self):
        # 🌙 CORE INDICATORS 🌙
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
        
        # Bollinger Band Width
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width_avg = self.I(talib.SMA, bb_width, timeperiod=self.bb_period)
        
        # Liquidation Zones (Proxy using swing highs/lows) ✨
        self.high_cluster = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.low_cluster = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Volatility and Volume
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period)
        
    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # 🌙 MOON DEV DEBUGGING 🌙
        print(f"\n🌙 Moon Dev Status Update 🌙")
        print(f"Price: {current_close:.2f} | Vol: {current_volume:.2f}")
        print(f"BB Width: {(self.bb_width_avg[-1] if self.bb_width_avg[-1] else 0):.4f}")
        print(f"High Cluster: {self.high_cluster[-1]:.2f} | Low Cluster: {self.low_cluster[-1]:.2f}")
        
        # Skip if indicators not ready
        if not all([self.bb_upper[-1], self.high_cluster[-1], self.atr[-1]]):
            return
            
        # ENTRY LOGIC 🚀
        if not self.position:
            # Long Entry: Close above high cluster + BB contraction
            if (current_close > self.high_cluster[-1] and 
                self.bb_width_avg[-1] < np.nanmean(self.bb_width_avg)):
                
                # Risk management 🌙
                stop_loss = self.low_cluster[-1] - 0.5 * self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_distance = abs(current_close - stop_loss)
                
                position_size = int(round(risk_amount / risk_distance))
                position_size = min(position_size, int(self.equity // current_close))
                
                if position_size > 0:
                    print(f"🚀 LONG SIGNAL TRIGGERED 🚀")
                    print(f"Entry: {current_close:.2f} | SL: {stop_loss:.2f}")
                    self.buy(size=position_size, sl=stop_loss, 
                           tags={'entry_type': 'cluster_breakout'})
            
            # Short Entry: Close below low cluster + BB contraction
            elif (current_close < self.low_cluster[-1] and 
                self.bb