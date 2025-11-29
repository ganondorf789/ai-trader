```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import os

# 🌙 MOON DEV DATA PREPARATION 🌙
def prepare_data(filepath):
    # Load and clean data
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

# 🌙 CUSTOM INDICATOR HELPERS 🌙
def bb_upper(close):
    upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5, matype=0)
    return upper

def bb_lower(close):
    _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5, matype=0)
    return lower

class SqueezeSurge(Strategy):
    risk_pct = 0.01  # 🌙 1% risk per trade
    
    def init(self):
        # 🌙 CORE INDICATORS 🌙
        self.ema2 = self.I(talib.EMA, self.data.Close, timeperiod=2)
        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)
        self.bb_width = self.I(lambda x: x - y, self.bb_upper, self.bb_lower)  # Width calculation
        self.bb_squeeze = self.I(talib.MIN, self.bb_width, timeperiod=20)
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, timeperiod=50)
        
    def next(self):
        # 🌙 MOON DEV SAFETY CHECKS 🌙
        if len(self.data) < 50:
            return
            
        # 🌙 CURRENT VALUES 🌙
        price = self.data.Close[-1]
        ema_current = self.ema2[-1]
        ema_prev = self.ema2[-2]
        
        # 🌙 SIGNAL CONDITIONS 🌙
        squeeze_condition = self.bb_width[-1] <= self.bb_squeeze[-1]
        volume_surge = self.data.Volume[-1] > self.volume_sma50[-1]
        
        # 🚀 LONG SIGNAL 🌙
        if not self.position.is_long and ema_current > ema_prev and squeeze_condition and volume_surge:
            sl = price * 0.99
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / (price - sl)))
            if position_size > 0:
                self.buy(size=position_size, sl=sl, tp=price*1.01)
                print(f"🚀🌙✨ LONG SURGE! Entry: {price:.2f} | Size: {position_size} | Moon Power: {self.bb_width[-1]:.2f}")
        
        # 🌙 SHORT SIGNAL 🌙
        elif not self.position.is_short and ema_current < ema_prev and squeeze_condition and volume_surge:
            sl = price * 1.01
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / (sl - price)))
            if position_size > 0:
                self.sell(size=position_size, sl=sl, tp=price*0.99)
                print(f"🌙🚀✨ SHORT SQUEEZE! Entry: {price:.2f} | Size: {position_size} | Moon Pressure: {self.bb_width[-1]:.2f}")
        
        # 🌙 EXIT CONDITIONS 🌙
        if self.position.is_long and ema_current < ema_prev:
            self.position.close()
            print(f"🌙💤 Closing LONG at {price:.2f} | EMA Reversal Detected")
        elif self.position.is_short and ema_current > ema_prev:
            self.position.close()
            print(f"🌙💤 Closing SHORT at {price:.2f} | EMA