```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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
                    print(f"✨ Updated long SL to {new_sl:.2f}")
            else:
                current_low = self.data.Low[-1]
                if current_low < trade.lowest_low:
                    trade.lowest_low = current_low
                new_sl = trade.lowest_low + self.atr[-1] * self.atr_multiplier
                if new_sl < trade.sl:
                    trade.sl = new_sl
                    print(f"✨ Updated short SL to {new_sl:.2f}")

        # Check exit signals
        if self.position.is_long and crossover(self.vi_m[-3:], self.vi_p[-3:]):
            print("🌙 VI+ crossed below VI-! Closing long!")
            self.position.close()
        elif self.position.is_short and crossover(self.vi_p[-3:], self.vi_m[-3:]):
            print("🌙 VI- crossed below VI+! Closing short!")
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
            price_div = self.data.Low[i] < self.data.Low[i-3] and self.cmo[i] > self.cmo[i-3]
            volume_ok = self.data.Volume[i] > self.volume_sma[i] and not np.isnan(self.volume_sma[i])
            if vi_cross and price_div and volume_ok:
                long_trigger = True
                break

        for offset