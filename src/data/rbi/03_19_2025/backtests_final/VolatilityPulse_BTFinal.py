I'll help fix the code by completing the short entry logic and ensuring proper position sizing. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# 🌙 Moon Dev Data Preparation Ritual ✨
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

class VolatilityPulse(Strategy):
    risk_per_trade = 0.02  # 🌙 2% Moon Dust Risk per Trade
    
    def init(self):
        # 🌙✨ Cosmic Indicators Setup
        # Bollinger Bands with Star Alignment 🌌
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close, name='UpperBB')
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close, name='LowerBB')
        self.bb_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band, name='BB_Width')
        self.min_bb_width = self.I(talib.MIN, self.bb_width, 20, name='Min_BB_Width')
        
        # 🌀 Chande Momentum Vortex
        self.cmo = self.I(talib.CMO, self.data.Close, 14, name='CMO')
        self.cmo_signal = self.I(talib.SMA, self.cmo, 9, name='CMO_Signal')
        
        self.entry_bar = 0  # 🌑 Moon Phase Tracker

    def next(self):
        # 🌑🌕 Lunar Position Management
        if self.position:
            current_bar = len(self.data) - 1
            if (current_bar - self.entry_bar) >= 10:
                self.position.close()
                print(f"🌙 Moon Dev Time Exit! 🕒 Closed at {self.data.Close[-1]}")

        else:
            # 🌟✨ Celestial Entry Conditions
            if np.isnan(self.upper_band[-1]) or np.isnan(self.cmo[-1]):
                return

            # 🌌 Cosmic Contraction Check
            bb_contraction = self.bb_width[-1] <= self.min_bb_width[-1]
            
            # 🌀 Momentum Alignment
            cmo_up = (self.cmo[-2] < self.cmo_signal[-2]) and (self.cmo[-1] > self.cmo_signal[-1])
            cmo_down = (self.cmo[-2] > self.cmo_signal[-2]) and (self.cmo[-1] < self.cmo_signal[-1])
            
            # 🚀 Long Entry Protocol
            if bb_contraction and cmo_up and self.data.Close[-1] > self.upper_band[-1]:
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.lower_band[-1]
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    take_profit = self.data.Close[-1] + (self.upper_band[-1] - self.lower_band[-1])
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    self.entry_bar = len(self.data) - 1
                    print(f"🌙🚀 LUNAR LIFT OFF! Long {position_size} @ {self.data.Close[-1]} SL: {stop_loss} TP: {take_profit}")

            # 🌪️ Short Entry Protocol
            elif bb_contraction and cmo_down and self.data.Close[-1] < self.lower_band[-1]:
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.upper_band[-1]
                risk_per_share = stop_loss - self.data.Close[-1]
                
                if risk_per_share > 0:
                    position_size =