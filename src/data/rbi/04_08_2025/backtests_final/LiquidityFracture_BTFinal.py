Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
print("🌙 Loading celestial market data...")
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
print("🌌 Data aligned with lunar cycles successfully!")

class LiquidityFracture(Strategy):
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    fib_level = 0.618
    risk_pct = 0.01
    volume_sma_period = 20
    lookback_period = 100

    def init(self):
        # Bollinger Bands components
        self.upper_band = self.I(self._upper_bb, self.data.Close, self.bb_period, self.bb_dev)
        self.lower_band = self.I(self._lower_bb, self.data.Close, self.bb_period, self.bb_dev)
        
        # Swing high/low detection
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Volume indicator
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_sma_period)
        
        # Bandwidth tracking
        self.bb_width_history = []
        print("🌠 Strategy indicators initialized for cosmic analysis...")

    def _upper_bb(self, close, period, dev):
        sma = talib.SMA(close, timeperiod=period)
        std = talib.STDDEV(close, timeperiod=period)
        return sma + dev * std

    def _lower_bb(self, close, period, dev):
        sma = talib.SMA(close, timeperiod=period)
        std = talib.STDDEV(close, timeperiod=period)
        return sma - dev * std

    def next(self):
        if len(self.data) < max(self.swing_period, self.lookback_period):
            return

        # Calculate Bollinger Band contraction
        current_width = self.upper_band[-1] - self.lower_band[-1]
        self.bb_width_history.append(current_width)
        if len(self.bb_width_history) > self.lookback_period:
            self.bb_width_history.pop(0)

        if len(self.bb_width_history) < self.lookback_period:
            return

        # Calculate width percentile
        percentile = (np.sum(np.array(self.bb_width_history) < current_width) / len(self.bb_width_history) * 100

        # Moon Dev debug prints 🌙
        print(f"🌙 Current BB Width: {current_width:.2f} | Percentile: {percentile:.1f}%")

        # Entry logic
        if not self.position:
            # Long entry conditions
            if (self.data.Low[-1] == self.swing_low[-1] and 
                percentile < 20 and 
                self.data.Close[-1] > self.upper_band[-1] and 
                self.data.Volume[-1] > self.volume_sma[-1]):
                
                swing_low_extreme = self.swing_low[-1]
                stop_loss = min(swing_low_extreme, self.lower_band[-1])
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (self.data.Close[-1] - stop_loss)))
                
                if position_size > 0:
                    fib_target = self.data.Close[-1] + (self.data.Close[-1] - swing_low_extreme) * self.fib_level
                    self.buy(size=position_size, sl=stop_loss, tp=fib_target)
                    print(f"🚀🌙 LONG ENTRY! Size: {position_size} | SL: {stop_loss:.2f} | TP: {fib_target:.