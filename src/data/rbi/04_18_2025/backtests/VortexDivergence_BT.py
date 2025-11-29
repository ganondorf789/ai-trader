```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
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

class VortexDivergence(Strategy):
    risk_percent = 0.01
    
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        print("🌙 Initializing Moon Dev Vortex Divergence Strategy...")
        
        # Vortex Indicator
        def compute_vortex(high, low, close, window):
            vi = ta.vortex(high=high, low=low, close=close, window=window)
            return vi[f'VIp_{window}'], vi[f'VIm_{window}']
        self.vi_plus, self.vi_minus = self.I(compute_vortex, self.data.High, self.data.Low, self.data.Close, 14, name='Vortex')

        # OBV
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')

        # Keltner Channel
        def compute_kc(high, low, close, ema_period, atr_period, multiplier):
            ema = talib.EMA(close, ema_period)
            atr = talib.ATR(high, low, close, atr_period)
            return ema + multiplier*atr, ema, ema - multiplier*atr
        self.upper_kc, self.middle_kc, self.lower_kc = self.I(compute_kc, self.data.High, self.data.Low, self.data.Close, 20, 20, 2, name='KC')

        # Swing detection
        self.swing_high = self.I(talib.MAX, self.data.High, 5, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, 5, name='SwingLow')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')

        # Track previous swings
        self.prev_swing_high_price = None
        self.prev_swing_high_obv = None
        self.prev_swing_low_price = None
        self.prev_swing_low_obv = None

    def next(self):
        price = self.data.Close[-1]
        print(f"\n🌙 Processing candle at {self.data.index[-1]} | Price: {price:.2f}")

        # Skip if already in position
        if self.position:
            return

        # Long entry logic 🚀
        if crossover(self.vi_plus, self.vi_minus):
            print("✨ VI+ crossed VI- (Bullish)")
            current_low = self.data.Low[-1]
            swing_low = self.swing_low[-1]
            
            if (current_low == swing_low and 
                self.prev_swing_low_price and
                current_low < self.prev_swing_low_price and
                self.obv[-1] > self.prev_swing_low_obv):
                
                print(f"✅ Bullish divergence detected! Price LL: {current_low:.2f} < {self.prev_swing_low_price:.2f}, OBV HL: {self.obv[-1]:.2f} > {self.prev_swing_low_obv:.2f}")
                
                if abs(price - self.middle_kc[-1])/self.middle_kc[-1] <= 0.005:
                    print("🎯 Price at Keltner midpoint")
                    self.enter_long()

        # Update swing lows
        if self.data.Low[-1] == self.swing_low[-1]:
            self.prev_swing_low_price = self.data.Low[-1]
            self.prev_swing_low_obv = self.obv[-1]

        # Short entry logic 📉
        if crossover(self.vi_minus, self.vi_plus):
            print("✨ VI- crossed