Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data 🌙
print("🌙✨ Moon Dev Data Loading System Activated...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print("🌙 Data successfully cleansed and aligned with lunar cycles!")

class DeltaClusterReversal(Strategy):
    risk_pct = 0.01
    rsi_period = 14
    swing_window = 20
    atr_period = 14
    vol_roc_window = 3

    def init(self):
        print("🌙✨ Initializing Moon Dev Trading Algorithms...")
        # Core indicators 🌙✨
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, 5)
        
        # Volume acceleration system 🌊
        self.vol_roc = self.I(lambda v: v.pct_change(self.vol_roc_window)*100, self.data.Volume)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20)
        print("🌙 Indicators calibrated to lunar gravitational patterns!")

    def next(self):
        if len(self.data) < 50:  # Warmup period
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]

        # Moon Dev exit system 🌙💨
        if self.position and (atr < atr_ma):
            self.position.close()
            print(f"🌙✨ Volatility exit! ATR({atr:.2f}) < MA({atr_ma:.2f})")

        # Liquidation cluster detection (simulated) 🌪️
        is_bullish_cluster = (self.data.Low[-1] <= self.swing_low[-1]) and \
                            (self.data.Volume[-1] > self.vol_ma[-1])
        
        is_bearish_cluster = (self.data.High[-1] >= self.swing_high[-1]) and \
                            (self.data.Volume[-1] > self.vol_ma[-1])

        # RSI divergence detection 🔄
        bull_div = (self.data.Low[-1] < self.data.Low[-2]) and \
                  (self.rsi[-1] > self.rsi[-2])
        
        bear_div = (self.data.High[-1] > self.data.High[-2]) and \
                  (self.rsi[-1] < self.rsi[-2])

        # Volume acceleration filter 📉
        vol_slowing = self.vol_roc[-1] < self.vol_roc[-2]

        # Moon-sized risk management 🌙💸
        if not self.position:
            risk_amount = self.equity * self.risk_pct
            
            # Long entry constellation 🌟
            if is_bullish_cluster and bull_div and vol_slowing:
                sl = self.swing_low[-1] - 0.5*atr
                size = int(round(risk_amount / (price - sl)))
                if size > 0: 
                    self.buy(size=size, sl=sl)
                    print(f"🚀🌙 BULLISH REVERSAL! Size: {size} | Risk: {self.risk_pct*100}%")

            # Short entry constellation 🌑
            elif is_bearish_cluster and bear_div and vol_slowing:
                sl = self.swing_high[-1] + 0.5*atr