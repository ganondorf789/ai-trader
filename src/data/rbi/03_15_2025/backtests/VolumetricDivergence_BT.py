Here's the complete backtesting implementation for the VolumetricDivergence strategy:

```python
# 🌙 Moon Dev's Volumetric Divergence Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# =====================
# DATA PREPARATION ✨
# =====================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌙✨
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# CORE STRATEGY 🚀
# =====================
class VolumetricDivergence(Strategy):
    atr_period = 14
    swing_window = 20
    risk_pct = 0.01  # 1% risk per trade
    size = 1_000_000  # Account size
    
    def init(self):
        # 🌙✨ Calculate indicators using TA-Lib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        
        # Price swing calculations 🌙
        self.price_highs = self.I(talib.MAX, self.data.High, self.swing_window)
        self.price_lows = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # ATR swings 🌙
        self.atr_highs = self.I(talib.MAX, self.atr, self.swing_window)
        self.atr_lows = self.I(talib.MIN, self.atr, self.swing_window)
        
        # OBV swings ✨
        self.obv_highs = self.I(talib.MAX, self.obv, self.swing_window)
        self.obv_lows = self.I(talib.MIN, self.obv, self.swing_window)
        
        self.trade_active = False
        print("🌙✨ Moon Dev Indicators Initialized! ✨🌙")

    def next(self):
        if len(self.data) < self.swing_window * 2:
            return
            
        # 🌙 Current swing values
        curr_ph = self.price_highs[-1]
        prev_ph = self.price_highs[-self.swing_window]
        curr_pl = self.price_lows[-1]
        prev_pl = self.price_lows[-self.swing_window]
        
        curr_ah = self.atr_highs[-1]
        prev_ah = self.atr_highs[-self.swing_window]
        curr_al = self.atr_lows[-1]
        prev_al = self.atr_lows[-self.swing_window]
        
        curr_oh = self.obv_highs[-1]
        prev_oh = self.obv_highs[-self.swing_window]
        curr_ol = self.obv_lows[-1]
        prev_ol = self.obv_lows[-self.swing_window]

        # 🌙✨ Divergence Detection
        bearish_div = (curr_ph > prev_ph) and (curr_ah < prev_ah) and (curr_oh < prev_oh)
        bullish_div = (curr_pl < prev_pl) and (curr_al > prev_al) and (curr_ol > prev_ol)
        
        if not self.position:
            if bearish_div:
                self.enter_short()
            elif bullish_div:
                self.enter_long()
        else:
            self.manage_trade()

    def enter_long(self):
        atr = self.atr[-1]
        if atr == 0:
            return
            
        # 🌙 Risk management calculations
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = 1.5 * atr
        position_size = int(round(risk_amount / risk_per_unit))
        
        if position_size == 0:
            return
            
        entry_price = self.data.Close[-1]
        stop_loss = entry_price