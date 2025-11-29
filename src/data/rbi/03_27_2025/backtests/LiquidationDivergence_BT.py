Here's the complete backtesting implementation for the LiquidationDivergence strategy:

```python
# 🌙 Moon Dev Backtest AI Implementation 🚀
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Data preparation with Moon Dev standards 🌕
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Cleanse and prepare cosmic data 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationDivergence(Strategy):
    # Galactic parameters 🌠
    liq_cluster_window = 20
    atr_period = 14
    funding_std_threshold = 2
    risk_percent = 0.02
    rr_ratio = 2
    daily_loss_limit = 0.05
    max_consecutive_losses = 3

    def init(self):
        # Stellar indicator calculations 🌟
        self.swing_high = self.I(talib.MAX, self.data.High, self.liq_cluster_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.liq_cluster_window)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Funding rate divergence simulation 🌗
        self.funding_ma_fast = self.I(talib.SMA, self.data.Close, 8)
        self.funding_ma_slow = self.I(talib.SMA, self.data.Close, 21)
        self.funding_spread = self.I(lambda x,y: x-y, self.funding_ma_fast, self.funding_ma_slow)
        self.funding_std = self.I(talib.STDDEV, self.funding_spread, 20)
        self.funding_z = self.I(lambda x,y: x/y, self.funding_spread, self.funding_std)
        
        # Cosmic volume forces 📈
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.volume_ratio = self.I(lambda x,y: x/y, self.data.Volume, self.volume_ma)
        
        # Lunar tracking variables 🌑
        self.consecutive_losses = 0
        self.daily_equity = [self._equity]

    def next(self):
        # Check orbital conditions 🛰️
        if len(self.data) < 50:  # Ensure enough cosmic history
            return

        # Cosmic risk management 🌍
        current_day = self.data.index[-1].date()
        if current_day != pd.Timestamp(self.daily_equity[-1][0]).date():
            daily_return = (self.equity - self.daily_equity[-1][1]) / self.daily_equity[-1][1]
            if daily_return < -self.daily_loss_limit:
                print(f"🌚 Emergency! Daily loss limit breached: {daily_return:.2%}")
                return
            self.daily_equity.append((current_day, self.equity))

        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"🌘 Cooling period active ({self.consecutive_losses} consecutive losses)")
            return

        # Planetary alignment conditions 🌐
        price = self.data.Close[-1]
        upper_cluster = self.swing_high[-1]
        lower_cluster = self.swing_low[-1]
        atr = self.atr[-1]
        
        # Long entry constellation 🌠
        long_cond = (
            (price < lower_cluster) and
            (abs(self.funding_z[-1]) >= self.funding_std_threshold) and
            (self.volume_ratio[-1] >= 1.5) and
            (self.data.Close[-1] > (self.data.High[-1] + self.data.Low[-1]) / 2)
            
        if long_cond and not self.position:
            # Calculate meteorite-proof position size ☄️
            sl = lower_cluster - atr
            risk_per_share = price - sl
            position_size = (self.risk