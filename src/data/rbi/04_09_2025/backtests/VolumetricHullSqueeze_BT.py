To address the user's request, here's the implementation of the VolumetricHullSqueeze strategy for backtesting.py:

```python
import pandas as pd
import talib
import pandas_ta as pdta
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricHullSqueeze(Strategy):
    hma_period = 10
    bb_length = 20
    bb_std_dev = 2
    atr_period = 14
    risk_pct = 0.01
    profit_multiplier = 2
    trail_multiplier = 1.5
    max_bars_held = 5 * 24 * 4  # 5 days in 15m bars
    blackout_bars = 3 * 24 * 4  # 3 days in 15m bars

    def init(self):
        # Trend components 🌙
        self.I(pdta.hma, self.data.Close, length=self.hma_period, name='HMA')
        self.I(talib.LINEARREG_SLOPE, self.data.HMA, timeperiod=5, name='HMA_SLOPE')
        
        # Volatility components ✨
        self.I(pdta.vwma, self.data.Close, self.data.Volume, length=self.bb_length, name='VWMA')
        self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_length, name='BB_STDDEV')
        self.I(lambda _: self.data.VWMA + self.bb_std_dev * self.data.BB_STDDEV, name='UPPER_BAND')
        self.I(lambda _: self.data.VWMA - self.bb_std_dev * self.data.BB_STDDEV, name='LOWER_BAND')
        self.I(lambda _: (self.data.UPPER_BAND - self.data.LOWER_BAND) / self.data.VWMA, name='BB_WIDTH')
        
        # Volume confirmation 📊
        self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_MA20')
        self.I(lambda _: (self.data.Volume / self.data.VOL_MA20) * 100, name='VOL_OSC')
        
        # Risk management components ⚠️
        self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # On-chain reserves (placeholder) 🔗
        self.I(talib.SMA, self.data.Close, timeperiod=7, name='RESERVE_MA7')  # Placeholder
        
        self.last_exit_bar = 0
        self.entry_bar = 0

    def next(self):
        current_bar = len(self.data) - 1
        
        # Moon-themed debug prints 🌕
        print(f"\n🌙 Processing {self.data.index[-1]} - Close: {self.data.Close[-1]:.2f}")
        print(f"HMA Slope: {self.data.HMA_SLOPE[-1]:.4f} | BB Width: {self.data.BB_WIDTH[-1]:.4f}")
        print(f"Volume Osc: {self.data.VOL_OSC[-1]:.1f}% | ATR: {self.data.ATR[-1]:.2f}")

        # Exit conditions 🚪
        if self.position:
            # Trailing stop calculation 🛑
            trail_price = self.data.High[-1] - self.trail_multiplier * self.data.ATR[-1]
            self.position.sl = max(self.position.sl or -np.inf, trail_price)

            # Profit target 🎯
            if self.data.Close[-1] >= self.position.entry_price + self.profit_multiplier * self.data.ATR[-1]:
                print(f"🚀✨ PROFIT TARGET HIT! Closing position at {self.data.Close[-1