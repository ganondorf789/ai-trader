I've analyzed the code and found a few technical issues to fix while preserving the strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and clean data with Moon Dev precision 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and prepare columns with lunar accuracy 🌕
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Moon Dev validation checks 🚀
assert 'funding_rate' in data.columns, "🌙 CRITICAL: Missing funding_rate column - required for lunar cycle analysis!"
assert 'long_short_ratio' in data.columns, "🌙 CRITICAL: Missing long_short_ratio column - essential for cosmic positioning!"

class ExtremeSentimentShort(Strategy):
    risk_pct = 0.01  # 1% per trade (Moon Dev recommended risk) 🌙
    fr_period = 2880  # 30 days in 15m intervals (30*24*4) - lunar cycle optimized
    ls_percentile = 90  # Cosmic percentile threshold ✨
    bb_length = 20  # Standard Bollinger Band length
    bb_dev = 2  # 2 standard deviations - cosmic expansion factor 🌌
    max_bars = 96  # 24 hours in 15m (one Earth day) 🌍

    def init(self):
        # Funding Rate Indicators with lunar precision 🌙
        self.fr_ma = self.I(talib.SMA, self.data.funding_rate, self.fr_period)
        self.fr_std = self.I(talib.STDDEV, self.data.funding_rate, self.fr_period)
        self.fr_threshold = self.I(lambda ma, std: ma + 2*std, self.fr_ma, self.fr_std)

        # Long/Short Ratio Analysis with cosmic alignment ✨
        self.ls_pct = self.I(ta.percentile, self.data.long_short_ratio, self.fr_period, self.ls_percentile)

        # Bollinger Bands Setup with gravitational precision 🌕
        self.lower_bb = self.I(talib.BBANDS, self.data.Close, self.bb_length, self.bb_dev, self.bb_dev)[2]

        self.entry_bar = 0  # Track position duration (lunar cycles) 🌑

    def next(self):
        # Wait for indicators to warm up (cosmic alignment period) 🌌
        if len(self.data) < self.fr_period:
            print("🌙 Waiting for cosmic indicators to align...")
            return

        # Get current values with lunar precision 🌙
        current_fr = self.data.funding_rate[-1]
        fr_threshold = self.fr_threshold[-1]
        current_ls = self.data.long_short_ratio[-1]
        ls_pct = self.ls_pct[-1]
        current_close = self.data.Close[-1]

        # Moon Dev Galactic Debug Console 🖥️
        print(f"🌙 Funding Rate: {current_fr:.6f} | Cosmic Threshold: {fr_threshold:.6f}")
        print(f"✨ L/S Ratio: {current_ls:.2f} | 90th Cosmic Percentile: {ls_pct:.2f}")

        # Entry Logic - Cosmic Short Signal 🚀
        if not self.position:
            if current_fr > fr_threshold and current_ls >= ls_pct:
                print(f"🚀🚀🚀 MOON DEV COSMIC SHORT ACTIVATED! Selling at {current_close:.2f} 🚀🚀🚀")
                self.sell(size=self.risk_pct)
                self.entry_bar = len(self.data)

        # Exit Logic - Lunar Profit Taking 🌙
        if self.position:
            # Bollinger Band Exit - Gravitational Pull 🎯
            if self.data.Low[-1] <= self.lower_bb[-1]:
                print(f"🌙✨ COSMIC PROFIT TAKEN! Closing at {current_close:.2f}")
                self.position.close()
            
            # Time-based Exit - Earth Rotation Protection 🌍
            elif len