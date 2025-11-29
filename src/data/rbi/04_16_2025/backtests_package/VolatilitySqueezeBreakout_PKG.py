Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper Moon Dev implementations 🌙✨:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLATILITY SQUEEZE BREAKOUT STRATEGY

import pandas as pd
import talib
from backtesting import Backtest, Strategy
import datetime
import numpy as np

# =====================
# DATA PREPARATION
# =====================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY IMPLEMENTATION
# =====================
class VolatilitySqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    max_concurrent_trades = 3
    open_trades = 0

    def init(self):
        # =====================
        # INDICATOR CALCULATION 🌙
        # =====================
        # Bollinger Bands
        self.upper_bb = self.I(talib.BBANDS, self.data.Close, 
                              timeperiod=20, nbdevup=2, nbdevdn=2, 
                              matype=0, name='Upper BB', which=0)
        self.lower_bb = self.I(talib.BBANDS, self.data.Close,
                              timeperiod=20, nbdevup=2, nbdevdn=2,
                              matype=0, name='Lower BB', which=2)

        # Keltner Channel
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.middle_kc = self.I(talib.EMA, typical_price, 20)
        atr20 = self.I(talib.ATR, self.data.High, self.data.Low, 
                       self.data.Close, 20)
        self.upper_kc = self.I(lambda mid, atr: mid + atr*1.5, 
                              self.middle_kc, atr20)
        self.lower_kc = self.I(lambda mid, atr: mid - atr*1.5,
                              self.middle_kc, atr20)

        # Volume and Momentum
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.macd_hist = self.I(lambda close: talib.MACD(close)[2],
                               self.data.Close)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low,
                           self.data.Close, 14)

        # Squeeze detection
        self.bb_width = self.I(lambda u, l: u - l, self.upper_bb, self.lower_bb)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, 20)

        self.squeeze_count = 0  # Track consecutive squeeze bars

    def next(self):
        # =====================
        # TIME FILTER 🌓
        # =====================
        current_time = self.data.index[-1].time()
        if (datetime.time(0, 0) <= current_time < datetime.time(0, 30) or \
           current_time >= datetime.time(23, 30)):
            print("🌙 Moon Dev Debug: Outside trading hours - skipping")
            return

        # =====================
        # INDICATOR VALUES 🌙
        # =====================
        close = self.data.Close[-1]
        upper_bb = self.upper_bb[-1]
        lower_bb = self.lower_bb[-1]
        upper_kc = self.upper_kc[-1]
        lower_kc = self.lower_kc[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        macd_hist = self.macd_hist[-1]
        prev_macd_hist = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0
        atr14 = self.atr14[-1]

        # =