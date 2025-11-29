I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Liquidation Squeeze Backtest 🌙
import pandas as pd
import talib
from backtesting import Strategy, Backtest

# =====================
# DATA PREPARATION 🌌
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse cosmic dust from column names 🪐
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align celestial coordinates 🌠
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert to proper time continuum ⏳
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS 🚀
# =====================
class LiquidationSqueeze(Strategy):
    # Cosmic configuration 🌑
    bb_period = 20
    bb_mult = 2
    atr_period = 14
    risk_pct = 0.01  # Fixed: converted to fraction for percentage sizing
    swing_period = 20
    volume_multiplier = 2

    def init(self):
        # Cosmic indicators 🌟
        # Bollinger Bands Constellation 🌌
        self.bb_middle = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period)
        self.bb_std = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_period)
        self.bb_upper = self.I(lambda m, s: m + self.bb_mult*s, self.bb_middle, self.bb_std)
        self.bb_lower = self.I(lambda m, s: m - self.bb_mult*s, self.bb_middle, self.bb_std)
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_period)
        
        # Volatility Meteor Shower ☄️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Gravity Wells (Swing Points) 🕳️
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Cosmic Energy Flow (Volume) 🌊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.bb_period)

    def next(self):
        # Avoid multiple cosmic alignments 🌘
        if self.position:
            return

        # Current celestial readings 🔭
        price = self.data.Close[-1]
        bbw = self.bb_width[-1]
        bbw_ma = self.bb_width_ma[-1]
        volume = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        # Cosmic Event Detection 🌑
        squeeze_activated = bbw < bbw_ma
        volume_surge = volume > self.volume_multiplier * vol_ma
        below_lower = price < self.bb_lower[-1]
        above_upper = price > self.bb_upper[-1]

        # Lunar Trade Alignment 🌕
        if squeeze_activated and volume_surge:
            # Long Entry: Black Hole Reversal 🌑➡️🌕
            if below_lower:
                sl = self.swing_low[-1] - self.atr[-1]
                risk = abs(price - sl)
                position_size = (self.equity * self.risk_pct) / risk  # Fixed: using percentage sizing
                tp = price + 3*risk  # 3:1 reward ratio
                
                if position_size > 0:
                    print(f"🌙✨ BLACK HOLE BREACH! Buying {position_size:.2f}% of equity