I'll fix the incomplete backtest code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
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

class VolatilitySynchron(Strategy):
    keltner_length = 20
    atr_multiplier = 2
    percentile_lookback = 50
    entropy_threshold = 80
    gap_threshold = 75
    exit_entropy_threshold = 50
    risk_percent = 0.01
    max_open_trades = 3

    def init(self):
        # Keltner Channel components
        self.ema = self.I(talib.EMA, self.data.Close, self.keltner_length)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.keltner_length)
        self.upper_band = self.ema + (self.atr * self.atr_multiplier)
        self.lower_band = self.ema - (self.atr * self.atr_multiplier)
        
        # Keltner Squeeze Normalization
        self.min_atr = self.I(talib.MIN, self.atr, self.percentile_lookback)
        self.max_atr = self.I(talib.MAX, self.atr, self.percentile_lookback)
        self.norm_keltner = (self.atr - self.min_atr) / (self.max_atr - self.min_atr + 1e-10)
        
        # Volume Entropy
        def entropy_calculation(series):
            series = series[series > 0]
            if len(series) < 1: return 0
            probs = series / series.sum()
            return -np.sum(probs * np.log(probs))
        
        self.volume_entropy = self.I(lambda: self.data.Volume.rolling(288).apply(entropy_calculation))
        self.entropy_pct = self.I(ta.percentile, self.volume_entropy, self.percentile_lookback, self.entropy_threshold)
        
        # Overnight Gap
        self.gap = self.I(lambda: (self.data.Open - self.data.Close.shift(1)) / self.data.Close.shift(1))
        self.gap_abs = self.I(lambda: abs(self.gap))
        self.gap_pct = self.I(ta.percentile, self.gap_abs, self.percentile_lookback, self.gap_threshold)
        
        # Exit indicators
        self.exit_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        print(f"🌙 Processing {self.data.index[-1]}...")
        print(f"✨ Current Equity: {self.equity:.2f}")
        
        if len(self.trades) >= self.max_open_trades:
            print("🚫 Max open trades reached - skipping new entries")
            return

        # Entry conditions
        current_norm = self.norm_keltner[-1]
        prev_norm = self.norm_keltner[-2] if len(self.norm_keltner) > 1 else 0
        keltner_signal = current_norm > 0.2 and prev_norm <= 0.2
        
        entropy_signal = self.entropy_pct[-1] >= self.entropy_threshold
        gap_signal = self.gap_pct[-1] >= self.gap_threshold
        
        # Long entry
        if keltner_signal and entropy_signal and gap_signal and self.gap[-1] > 0:
            atr_val = self.exit_atr[-1]
            sl_price = self.data.Close[-1] - (2 * atr_val)
            risk_amount = self.equity *