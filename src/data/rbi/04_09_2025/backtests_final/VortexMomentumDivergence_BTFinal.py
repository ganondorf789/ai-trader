I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns with proper case
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexMomentumDivergence(Strategy):
    vip_period = 14
    cmo_period = 20
    volume_sma_period = 20
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    atr_multiplier = 2

    def init(self):
        # Calculate indicators
        vi_p, vi_m = talib.VORTEX(self.data.High, self.data.Low, self.data.Close, self.vip_period)
        self.vi_p = self.I(lambda: vi_p, name='VI+')
        self.vi_m = self.I(lambda: vi_m, name='VI-')
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period, name='Volume SMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')

    def next(self):
        # Skip early bars without sufficient data
        if len(self.data) < max(self.vip_period, self.cmo_period, self.volume_sma_period, self.atr_period) + 4:
            return

        current_idx = len(self.data) - 1

        # Update trailing stops for existing trades
        for trade in self.trades:
            if trade.is_long:
                current_high = self.data.High[-1]
                if current_high > trade.highest_high:
                    trade.highest_high = current_high
                new_sl = trade.highest_high - self.atr[-1] * self.atr_multiplier
                if new_sl > trade.sl:
                    trade.sl = new_sl
                    print(f"🌙 Moon Dev Alert: Updated long SL to {new_sl:.2f} (Trailing Stop Active)")
            else:
                current_low = self.data.Low[-1]
                if current_low < trade.lowest_low:
                    trade.lowest_low = current_low
                new_sl = trade.lowest_low + self.atr[-1] * self.atr_multiplier
                if new_sl < trade.sl:
                    trade.sl = new_sl
                    print(f"🌙 Moon Dev Alert: Updated short SL to {new_sl:.2f} (Trailing Stop Active)")

        # Check exit signals
        if self.position.is_long and (self.vi_m[-2] < self.vi_p[-2] and self.vi_m[-1] > self.vi_p[-1]):
            print("🌙 Moon Signal: VI+ crossed below VI-! Closing long position!")
            self.position.close()
        elif self.position.is_short and (self.vi_p[-2] < self.vi_m[-2] and self.vi_p[-1] > self.vi_m[-1]):
            print("🌙 Moon Signal: VI- crossed below VI+! Closing short position!")
            self.position.close()

        # Check entry conditions
        long_trigger = False
        short_trigger = False

        # Check last 3 bars for entries
        for offset in range(3):
            i = current_idx - offset
            if i < 3:
                continue

            # Long conditions
            vi_cross = self.vi_p[i] > self.vi_m[i] and self.vi_p[i-1] <= self.vi_m[i-1]
            cmo_divergence = (self.data.Close[i