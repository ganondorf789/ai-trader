I'll help debug the code while maintaining the strategy logic. Here's the fixed version with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from collections import deque

# Data preprocessing
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

class FractalDivergence(Strategy):
    rsi_period = 14
    atr_period = 14
    fractal_window = 20
    risk_pct = 0.01
    tp_multiplier = 2
    sl_multiplier = 1
    
    def init(self):
        # Core Indicators 🌙
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Fractal Dimension Calculation ✨
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.fd = self.I(lambda high, low: talib.SUM(((high == talib.MAX(high, 5)) | (low == talib.MIN(low, 5))).astype(float), 20),
                         self.data.High, self.data.Low)
        
        # Volume Momentum 🚀
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, 5)
        
        # Swing tracking buffers
        self.swing_lows_price = deque(maxlen=2)
        self.swing_highs_price = deque(maxlen=2)
        self.swing_lows_rsi = deque(maxlen=2)
        self.swing_highs_rsi = deque(maxlen=2)

    def next(self):
        # Moon Dev Debug Header 🌙
        print(f"\n🌕 Moon Dev Strategy Cycle - {self.data.index[-1]} 🚀")
        print(f"💰 Equity: {self.equity:,.2f} | Price: {self.data.Close[-1]:.2f} 🌙")
        
        # Detect swing points ✨
        if self.data.Low[-1] == self.swing_low[-1]:
            self.swing_lows_price.append(self.data.Low[-1])
            print(f"🔻 Price Swing Low Detected: {self.data.Low[-1]:.2f}")
        if self.data.High[-1] == self.swing_high[-1]:
            self.swing_highs_price.append(self.data.High[-1])
            print(f"🔺 Price Swing High Detected: {self.data.High[-1]:.2f}")
        if self.rsi[-1] == talib.MIN(self.rsi, 5)[-1]:
            self.swing_lows_rsi.append(self.rsi[-1])
            print(f"📉 RSI Swing Low: {self.rsi[-1]:.2f}")
        if self.rsi[-1] == talib.MAX(self.rsi, 5)[-1]:
            self.swing_highs_rsi.append(self.rsi[-1])
            print(f"📈 RSI Swing High: {self.rsi[-1]:.2f}")

        # Divergence detection 🌙
        bullish_div = bearish_div = False
        if len(self.swing_lows_price) >= 2 and len(self.swing_lows_rsi) >= 2:
            if (self.swing_lows_price[-1] < self.swing_lows_price[-2] and
                self.swing_lows_rsi[-1] > self.swing_lows_rsi[-2]):
                bullish_div = True
                print(f"🚀 BULLISH DIVERGENCE CONFIRMED! Price: {self.swing_lows_price[-1]:.2f} < {self.swing_lows_price[-2]:.2f} | RSI: {self.swing_lows_rsi[-1]:.2f} > {self.swing_lows_rsi[-2]:.2f} 🌙")
                
        if len(self.swing_highs_price) >= 2 and len(self.swing_highs