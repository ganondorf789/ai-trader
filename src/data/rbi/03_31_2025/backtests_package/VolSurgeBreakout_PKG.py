Here's the fixed code with all indicator packages properly implemented and Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolSurgeBreakout Backtest Implementation ✨
# 🚀 Certified backtesting.lib FREE by Moon Dev Package AI 🌙

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
        # 🌗 KELTNER CHANNEL CALCULATION (Pure TA-Lib Implementation)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        # 🌙 Calculating Keltner Bands using numpy arrays
        self.keltner_upper = self.I(lambda: self.ema + (self.atr * self.keltner_multiplier), name='Keltner Upper')
        self.keltner_lower = self.I(lambda: self.ema - (self.atr * self.keltner_multiplier), name='Keltner Lower')
        
        # 📈 VOLUME INDICATOR (TA-Lib SMA)
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_avg_period, name='Vol Avg')
        
        print("🌕✨ Moon Dev Indicators Initialized Successfully! Ready for lunar trading cycles...")

    def next(self):
        # 🌑 MOON TRADE LOGIC
        if self.position:
            return  # No overlapping trades
            
        if len(self.data) < max(self.ema_period, self.atr_period, self.volume_avg_period):
            return  # Warmup period
            
        close = self.data.Close[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.volume_avg[-1]
        
        # 🌙 VOLUME SURGE CONDITION (250% increase - lunar gravity assist!)
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
        
        # 🌕 RISK MANAGEMENT CALCULATIONS (Lunar-grade precision)
        sl_price = entry_price - atr_val
        tp_price = entry_price + (2 * atr_val)
        risk_amount = equity * self.risk_percent
        position_size = int(round(risk_amount / atr_val))
        
        if position_size > 0:
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌕✨ MOONSHOT LONG! Entry: {entry_price:.2f} | Size: {position_size} BTC | 🛑 SL: {sl_price:.2f} | 🎯 TP: {tp_price:.2f}")
            print(f"🌙 Current Lunar Phase: Bullish Breakout | ATR: {atr_val:.2f} | Equity: ${equity:,.2f}")

    def enter_short(self):
        equity = self.broker.getvalue