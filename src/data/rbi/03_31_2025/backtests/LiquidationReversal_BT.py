Here's the complete implementation adhering to all specified requirements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and prepare data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    consecutive_losses = 0
    
    def init(self):
        # Core indicators ✨
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Liquidation High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Liquidation Low')
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI 14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR 20')
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume MA')
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 0.02, 0.2, name='Parabolic SAR')
        
        self.entry_level = None
        self.trade_direction = None

    def next(self):
        # Avoid overnight trading 🌙
        if self.data.index[-1].hour < 8 or self.data.index[-1].hour > 20:
            return
            
        # Check max consecutive losses
        if self.consecutive_losses >= 3:
            print(f"🌧️  Moon Dev Alert: 3 consecutive losses - Cooling off! 🌧️")
            return
            
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Entry Logic Cluster 🌙✨
        if not self.position:
            # Bearish setup 🐻
            if current_close > self.swing_high[-1]:
                if (self.rsi[-1] < self.rsi[-3] and 
                    current_volume > 2*self.vol_ma[-1]):
                    
                    move = current_high - self.swing_high[-1]
                    self.entry_level = current_high - move*0.382
                    self.trade_direction = 'short'
                    print(f"🌙 SHORT Signal! Swing High: {self.swing_high[-1]:.2f}")
                    print(f"   Entry Zone: {self.entry_level:.2f} ✨")

            # Bullish setup 🐮
            elif current_close < self.swing_low[-1]:
                if (self.rsi[-1] > self.rsi[-3] and 
                    current_volume > 2*self.vol_ma[-1]):
                    
                    move = self.swing_low[-1] - current_low
                    self.entry_level = current_low + move*0.382
                    self.trade_direction = 'long'
                    print(f"🌙 LONG Signal! Swing Low: {self.swing_low[-1]:.2f}")
                    print(f"   Entry Zone: {self.entry_level:.2f} ✨")

            # Execute entry if price retraces 🌊
            if self.trade_direction == 'short' and current_close <= self.entry_level:
                risk = abs(self.entry_level - (self.swing_high[-1] + self.atr[-1]))
                size = int(round((self.risk_pct * self.equity) / risk))
                self.sell(size=size, sl=self.swing_high[-1] + self.atr[-1],
                         tp=[self.entry_level - 1.5*self.atr[-1], self.swing_low[-1]])
                print(f"🚀 SHORT Entry! Size: {size} @