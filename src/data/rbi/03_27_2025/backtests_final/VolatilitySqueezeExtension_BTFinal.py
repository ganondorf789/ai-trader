Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing implementation:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Moon Dev Data Loading Sequence Initiated 🌙
print("\n=== MOON DEV DATA LOADING ===")
print("🌌 Scanning the cosmos for price data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print("✨ Data successfully aligned with lunar cycles!")

class VolatilitySqueezeExtension(Strategy):
    bb_period = 20
    bb_std = 2
    chaikin_fast = 3
    chaikin_slow = 10
    risk_pct = 0.01
    fib127 = 1.27
    fib1618 = 1.618

    def init(self):
        # Moon Dev Indicator Initialization Sequence 🌙
        print("\n=== INITIALIZING LUNAR INDICATORS ===")
        
        # Bollinger Bands
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[0], 
                              self.data.Close, name='UpperBB')
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[2], 
                              self.data.Close, name='LowerBB')
        print("🌗 Bollinger Bands calibrated to lunar gravity")
        
        # Bollinger Band Width
        self.bb_width = self.I(lambda: (self.upper_bb - self.lower_bb) / 
                              talib.SMA(self.data.Close, self.bb_period), name='BB_Width')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, self.bb_period, name='BB_Width_Avg')
        print("📏 Bandwidth measurements synchronized")
        
        # Chaikin Oscillator
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, self.data.Volume,
                             self.chaikin_fast, self.chaikin_slow, name='Chaikin')
        print("🌊 Chaikin tides measured")
        
        # Volatility measures
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.atr_sma = self.I(talib.SMA, self.atr, 20, name='ATR_SMA')
        print("⚡ Volatility sensors activated")
        
        # Swing lows
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        print("🏔️ Swing low detection online")

    def next(self):
        if self.position:
            return

        # Moon Dev Trade Detection Sequence 🌙
        window = 3
        valid_squeeze = any(self.bb_width[-i] < 0.3*self.bb_width_avg[-i] for i in range(1, window+1))
        valid_chaikin = any((self.chaikin[-i-2] < 0 and self.chaikin[-i-1] > 0) for i in range(1, window+1))
        valid_breakout = any(self.data.Close[-i] > self.upper_bb[-i] for i in range(1, window+1))

        if valid_squeeze and valid_chaikin and valid_breakout:
            # Moon Dev Trade Execution Protocol 🌙
            print("\n=== LUNAR TRADE DETECTED ===")
            print("🌑 Volatility squeeze confirmed")
            print("🌕 Chaikin momentum aligned")
            print("🚀 Breakout detected!")
            
            # Fibonacci calculations
            squeeze_high = max(self.data.High[-20:])
            squeeze_low = min(self.data.Low[-20:])
            fib_range = squeeze_high - squeeze_low
            
            entry_price