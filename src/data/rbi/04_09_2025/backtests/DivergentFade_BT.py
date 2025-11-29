```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Moon Dev Data Preparation 🌙
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping with Moon Dev precision ✨
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class DivergentFade(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    consecutive_losses = 0
    
    def init(self):
        # Moon Dev Indicators Setup 🌙
        self.upper_band = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, 0, name='Upper BB', which=0)
        self.middle_band = self.I(talib.SMA, self.data.Close, 20, name='Middle BB')
        self.lower_band = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, 0, name='Lower BB', which=2)
        
        # MACD Histogram with Moon Divergence Detection 🌙✨
        self.macd_hist = self.I(talib.MACD, self.data.Close, 12, 26, 9, name='MACD Hist', which=2)
        
        # ATR for Moon-Sized Stops 🌙🚀
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # Volume SMA for Fuel Check 🚀⛽
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 14, name='Volume SMA')
        
    def next(self):
        # Moon Dev Risk Force Field 🌙🛡️
        if self.consecutive_losses >= 3:
            print("🌙⚡ Moon Dev Circuit Breaker: 3 Consecutive Losses!")
            return
            
        if not self.position:
            # Short Entry Constellation 🌙💫
            if self.data.High[-1] > self.upper_band[-1]:
                if (self.data.Volume[-1] < self.data.Volume[-2] and 
                    self.data.Volume[-2] < self.data.Volume[-3]):
                    if (self.data.High[-1] > self.data.High[-2] and 
                        self.macd_hist[-1] < self.macd_hist[-2]):
                        
                        # Moon Position Sizing 🌙📐
                        stop_loss = self.data.Close[-1] + 1.5 * self.atr[-1]
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / (stop_loss - self.data.Close[-1])))
                        
                        if size > 0:
                            print(f"🌙🚀 Moon Dev Short Signal! Size: {size} | Entry: {self.data.Close[-1]:.2f}")
                            self.sell(size=size, sl=stop_loss, tag='short_entry')
            
            # Long Entry Constellation 🌙💫
            if self.data.Low[-1] < self.lower_band[-1]:
                if (self.data.Volume[-1] < self.data.Volume[-2] and 
                    self.data.Volume[-2] < self.data.Volume[-3]):
                    if (self.data.Low[-1] < self.data.Low[-2] and 
                        self.macd_hist[-1] > self.macd_hist[-2]):
                        
                        # Moon Position Sizing 🌙📐
                        stop_loss = self.data.Close[-1] - 1.5 * self.atr[-1]
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / (self.data.Close[-1] - stop_loss)))
                        
                        if size > 0:
                            print(f"🌙🚀 Moon Dev Long Signal! Size: {size} | Entry: {self.data.Close[-1]:.2f}")
                            self.buy