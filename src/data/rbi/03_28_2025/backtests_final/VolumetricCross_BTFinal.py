Here's the fixed and complete backtest code with Moon Dev's debug improvements:

```python
# 🌙 Moon Dev's VolumetricCross Backtest Script 🚀

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class VolumetricCross(Strategy):
    ema_period_fast = 50
    ema_period_slow = 200
    volume_ma_period = 20
    rsi_period = 14
    swing_period = 20
    risk_percent = 0.01  # 1% risk per trade

    def init(self):
        # 🌙 Moon Indicators Calculation
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period_fast, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period_slow, name='EMA200')
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, name='VOL_MA20')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI14')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW20')

    def next(self):
        # 🌙✨ Core Strategy Logic
        price = self.data.Close[-1]
        ema50_val = self.ema50[-1]
        ema200_val = self.ema200[-1]

        # 🚀 Entry Conditions
        if not self.position:
            # EMA crossover check (replaced backtesting.lib.crossover)
            ema_cross = (self.ema50[-2] <= self.ema200[-2] and 
                        self.ema50[-1] > self.ema200[-1])
            
            # Volume spike check
            volume_spike = (self.data.Volume[-1] > 
                           1.5 * self.volume_avg[-1])
            
            # Price above both EMAs
            price_above_emas = (price > ema50_val and 
                               price > ema200_val)

            if ema_cross and volume_spike and price_above_emas:
                # 🌙 Calculate stop loss levels
                swing_low_val = self.swing_low[-1]
                stop_loss_level = min(swing_low_val, ema200_val)
                
                # 🎯 Risk management calculations
                entry_price = price
                risk_amount = self.risk_percent * self.equity
                risk_per_unit = entry_price - stop_loss_level
                
                if risk_per_unit <= 0:
                    print("🌑 Moon Dev Warning: Invalid stop loss level detected! Aborting trade.")
                    return  # Avoid invalid SL
                
                position_size = int(round(risk_amount / risk_per_unit))
                take_profit = entry_price + 2 * (entry_price - stop_loss_level)

                # 🌙✨ Execute entry with Moon-themed print
                print(f"🚀 Moon Dev Long Signal! 🌙 EMA crossover + Volume spike detected!")
                print(f"📈 Entry: {entry_price:.2f} | SL: {stop_loss_level:.2f} | TP: {take_profit:.2f}")
                print(f"💎 Position Size: {position_size} units | Risk: {self.risk_percent*100}%")
                
                self.buy(size=position_size, 
                        sl=stop_loss_level,
                        tp=take_profit)

        # 🌕 Exit Condition - RSI Overbought
        elif self.position.is_long and self.rsi[-1] > 70:
            print(f"🌕 Moon Dev Exit Signal! RSI {self.rsi[-1]:.2f} > 70 - Taking profits!")
            self.position.close()

# 🌙 Data Preparation Magic
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Data Cleaning Ritual
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌌 Column Mapping to Backtesting.py standards
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ⏳ Proper datetime index setup
data['DateTime'] = pd