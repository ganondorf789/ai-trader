Here's the fully debugged and Moon Dev themed backtest code with all technical issues fixed while preserving the original strategy logic:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATIONBREAKOUT STRATEGY �✨

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION 🌙
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data according to Moon Dev standards ✨
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

# =====================
# STRATEGY IMPLEMENTATION 🌙
# =====================
class LiquidationBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    swing_period = 20      # Swing high/low detection period
    
    def init(self):
        # 🌙 Core Strategy Indicators
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING LOW')
        
        # 🌙 Volatility Indicators (4-hour ATR equivalent in 15m data)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                          timeperiod=16, name='ATR_4H')  # 16*15m = 4 hours
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=10, name='ATR_MA')
        
    def next(self):
        # Wait for indicators to warm up ✨
        if len(self.data) < 50:
            return
        
        current_price = self.data.Close[-1]
        equity = self.equity
        
        # 🌙 Volatility Contraction Check
        volatility_ok = self.atr[-1] < self.atr_ma[-1]
        
        # 🌙 Liquidation Zone Proximity (1% threshold)
        upper_zone = self.swing_high[-1]
        lower_zone = self.swing_low[-1]
        near_upper = current_price >= upper_zone * 0.99
        near_lower = current_price <= lower_zone * 1.01
        
        # =====================
        # ENTRY LOGIC 🌙🚀
        # =====================
        if not self.position and volatility_ok:
            # 🌙 Long Entry Condition
            if near_upper:
                sl_price = lower_zone * 0.999  # Stop loss below lower zone
                risk_per_share = abs(current_price - sl_price)
                position_size = (equity * self.risk_per_trade) / risk_per_share
                position_size = int(round(position_size))  # 🌙 Ensuring whole units
                
                if position_size > 0:
                    tp_price = current_price + (3 * risk_per_share)  # 3:1 RR
                    self.buy(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"🌙 MOON DEV LONG SIGNAL 🌕 | Entry: {current_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | Size: {position_size} ✨")
            
            # 🌙 Short Entry Condition
            elif near_lower:
                sl_price = upper_zone * 1.001  # Stop loss above upper zone
                risk_per_share = abs(sl_price - current_price)
                position_size = (equity * self.risk_per_trade) / risk_per_share
                position_size = int(round(position_size))  # 🌙 Ensuring whole units
                
                if position_size > 0:
                    tp_price = current_price - (3 * risk_per_share)  # 3:1 RR
                    self.sell(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"🌙 MOON