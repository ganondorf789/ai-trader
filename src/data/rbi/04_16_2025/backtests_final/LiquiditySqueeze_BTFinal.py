I'll fix the incomplete code while maintaining the original strategy logic and adding Moon Dev themed debug prints. Here's the complete fixed version:

```python
# 🌙 Moon Dev Backtest AI Implementation 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# ========== DATA PREPARATION ==========
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format column names 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# ========== LIQUIDITY SQUEEZE STRATEGY ==========
class LiquiditySqueeze(Strategy):
    # 🌙 Strategy Parameters
    bb_period = 20
    bb_dev = 2
    bbw_lookback = 100
    bbw_threshold = 20  # 20th percentile
    volume_ratio_threshold = 1.5
    risk_pct = 0.01  # 1% risk per trade
    swing_period = 20
    max_bars_held = 5

    def init(self):
        # 🌙✨ Bollinger Bands Calculation
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev, 0)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev, 0)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev, 0)[2], self.data.Close)
        
        # 🌙 BBW = (Upper - Lower)/Middle
        self.bbw = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        
        # 🌙 BBW Percentile Value (20th percentile)
        self.bbw_percentile_value = self.I(ta.percentile, self.bbw, self.bbw_lookback, q=self.bbw_threshold)
        
        # 🌙 Liquidation Heatmap Proxy (Volume Surge)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.volume_ratio = self.I(lambda v, sma: v/sma, self.data.Volume, self.volume_sma)
        
        # 🌙 Liquidity Pool Targets
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)

    def next(self):
        # 🌙✨ Debug Prints
        print(f"\n🌙 Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f}")
        print(f"   BBW: {self.bbw[-1]:.4f} < 20th% ({self.bbw_percentile_value[-1]:.4f})? {self.bbw[-1] < self.bbw_percentile_value[-1]}")
        print(f"   Vol Ratio: {self.volume_ratio[-1]:.2f}x SMA")

        if self.position:
            # 🌙 Time-based Exit
            if len(self.data) - self.position.entry_bar >= self.max_bars_held:
                print(f"   ⏳ Moon Dev Time Exit! Bars held: {len(self.data) - self.position.entry_bar}")
                self.position.close()
            return

        # 🌙 Entry Conditions
        entry_price = self.data.Close[-1]
        long_cond = (self.bbw[-1] < self.bbw_percentile_value[-1] and
                     entry_price > self.bb_upper[-1] and
                     self.volume_ratio[-1] > self.volume_ratio_threshold)
                     
        short_cond = (self.bbw[-1] < self.bbw_percentile