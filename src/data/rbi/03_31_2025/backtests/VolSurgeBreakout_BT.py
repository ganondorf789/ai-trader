```python
# 🌙 Moon Dev's VolSurgeBreakout Backtest Implementation ✨

import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# 🚀 DATA PREPARATION 
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSurgeBreakout(Strategy):
    ema_period = 20
    atr_period = 14
    keltner_multiplier = 2
    volume_avg_period = 10
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌗 KELTNER CHANNEL CALCULATION
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.keltner_upper = self.ema + (self.atr * self.keltner_multiplier)
        self.keltner_lower = self.ema - (self.atr * self.keltner_multiplier)
        
        # 📈 VOLUME INDICATOR
        self.volume_avg = self.I(talib.SMA, self.data.Volume, self.volume_avg_period, name='Vol Avg')

    def next(self):
        # 🌑 MOON TRADE LOGIC
        if self.position:
            return  # No overlapping trades
            
        if len(self.data) < max(self.ema_period, self.atr_period, self.volume_avg_period):
            return  # Warmup period
            
        close = self.data.Close[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.volume_avg[-1]
        
        # 🌙 VOLUME SURGE CONDITION (150% increase)
        vol_surge = volume >= (vol_avg * 2.5) if vol_avg else False
        
        # 🚀 LONG ENTRY: Breakout above Keltner Upper
        if close > self.keltner_upper[-1] and vol_surge:
            self.enter_long()
            
        # 🌑 SHORT ENTRY: Breakdown below Keltner Lower
        elif close < self.keltner_lower[-1] and vol_surge:
            self.enter_short()
    
    def enter_long(self):
        equity = self.broker.getvalue()
        atr_val = self.atr[-1]
        entry_price = self.data.Close[-1]
        
        # 🌕 RISK MANAGEMENT CALCULATIONS
        sl_price = entry_price - atr_val
        tp_price = entry_price + (2 * atr_val)
        risk_amount = equity * self.risk_percent
        position_size = int(round(risk_amount / atr_val))
        
        if position_size > 0:
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌕✨ MOONSHOT LONG! Entry: {entry_price:.2f} | Size: {position_size} BTC | 🛑 SL: {sl_price:.2f} | 🎯 TP: {tp_price:.2f}")

    def enter_short(self):
        equity = self.broker.getvalue()
        atr_val = self.atr[-1]
        entry_price = self.data.Close[-1]
        
        # 🌑 RISK MANAGEMENT CALCULATIONS
        sl_price = entry_price + atr_val
        tp_price = entry_price - (2 * atr_val)
        risk_amount = equity * self.risk_percent
        position_size = int(round(risk_amount / atr_val))
        
        if position_size > 0:
            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌑✨ CRESCENT SHORT! Entry: {entry_price:.2f} | Size: {position_size} BTC | 🛑 SL: {sl_price:.2f