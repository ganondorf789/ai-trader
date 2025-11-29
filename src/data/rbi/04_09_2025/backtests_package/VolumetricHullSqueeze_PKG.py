I'll fix the code by removing all `backtesting.lib` imports and replacing any related functions with proper alternatives. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as pdta
import numpy as np
from backtesting import Backtest, Strategy

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
        self.hma = self.I(pdta.hma, self.data.Close, length=self.hma_period, name='HMA')
        self.hma_slope = self.I(talib.LINEARREG_SLOPE, self.hma, timeperiod=5, name='HMA_SLOPE')
        
        # Volatility components ✨
        self.vwma = self.I(pdta.vwma, self.data.Close, self.data.Volume, length=self.bb_length, name='VWMA')
        self.bb_stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_length, name='BB_STDDEV')
        self.upper_band = self.I(lambda _: self.vwma + self.bb_std_dev * self.bb_stddev, name='UPPER_BAND')
        self.lower_band = self.I(lambda _: self.vwma - self.bb_std_dev * self.bb_stddev, name='LOWER_BAND')
        self.bb_width = self.I(lambda _: (self.upper_band - self.lower_band) / self.vwma, name='BB_WIDTH')
        
        # Volume confirmation 📊
        self.vol_ma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_MA20')
        self.vol_osc = self.I(lambda _: (self.data.Volume / self.vol_ma20) * 100, name='VOL_OSC')
        
        # Risk management components ⚠️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # On-chain reserves (placeholder) 🔗
        self.reserve_ma7 = self.I(talib.SMA, self.data.Close, timeperiod=7, name='RESERVE_MA7')  # Placeholder
        
        self.last_exit_bar = 0
        self.entry_bar = 0

    def next(self):
        current_bar = len(self.data) - 1
        
        # Moon-themed debug prints 🌕
        print(f"\n🌙 Processing {self.data.index[-1]} - Close: {self.data.Close[-1]:.2f}")
        print(f"HMA Slope: {self.hma_slope[-1]:.4f} | BB Width: {self.bb_width[-1]:.4f}")
        print(f"Volume Osc: {self.vol_osc[-1]:.1f}% | ATR: {self.atr[-1]:.2f}")

        # Exit conditions �
        if self.position:
            # Trailing stop calculation 🛑
            trail_price = self.data.High[-1] - self.trail_multiplier * self.atr[-1]
            self.position.sl = max(self.position.sl or -np.inf, trail_price)

            # Profit target 🎯
            if self.data.Close[-1] >= self