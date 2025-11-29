I'll fix the code by removing all `backtesting.lib` imports and replacing the crossover functions with proper implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Volatility Convergence Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========================
# 🌌 DATA PREPARATION
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse cosmic dust from column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align celestial coordinates 🌍
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# 🌠 VOLATILITY CONVERGENCE STRATEGY
# ========================
class VolatilityConvergence(Strategy):
    # Cosmic configuration ⚙️
    bb_period = 20
    bb_std_dev = 2
    rsi_period = 14
    risk_per_trade = 0.01  # 1% stardust risk per trade 🌌
    max_daily_loss = 0.05  # 5% supernova protection 💥

    def init(self):
        # 🌗 Core Orbital Indicators
        self.close = self.data.Close
        self.volume = self.data.Volume
        
        # 🌐 Synthetic Spread Calculation
        self.spread = self.I(lambda c: c - talib.SMA(c, self.bb_period), self.close)
        
        # 🪐 Bollinger Bands Configuration
        self.spread_mean = self.I(talib.SMA, self.spread, self.bb_period)
        self.spread_std = self.I(talib.STDDEV, self.spread, self.bb_period)
        self.upper_band = self.I(lambda m, s: m + self.bb_std_dev*s, self.spread_mean, self.spread_std)
        self.lower_band = self.I(lambda m, s: m - self.bb_std_dev*s, self.spread_mean, self.spread_std)
        
        # 📡 Volume Resonance Scanner
        self.volume_sma = self.I(talib.SMA, self.volume, 20)
        self.volume_ratio = self.I(lambda v, sma: v/sma, self.volume, self.volume_sma)
        
        # 🌀 Momentum Vortex Indicator
        self.rsi = self.I(talib.RSI, self.close, self.rsi_period)
        
        # 🛰️ Volatility Tracker
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.close, 14)
        
        # 🌑 Lunar Cycle Tracker
        self.daily_equity = [self.equity]

    def next(self):
        # 🚀 Interstellar Position Management
        current_price = self.close[-1]
        
        # 🌓 Check for supernova events (daily loss limit)
        if len(self.daily_equity) == 0 or self.data.index[-1].date() != pd.Timestamp(self.daily_equity[-1]).date():
            self.daily_equity.append(self.equity)
        daily_pnl = (self.equity - self.daily_equity[-1]) / self.daily_equity[-1]
        if daily_pnl <= -self.max_daily_loss:
            print(f"🌑💥 COSMIC MELTDOWN: Daily limit reached! Aborting at {self.data.index[-1]}")
            return

        if not self.position:
            # 🌌 Entry Protocol
            spread = self.spread[-1]
            vol_ratio = self.volume_ratio[-1]
            
            # 🌙 Moon Dev's Bullish Crossover Detection ✨
            long_signal = spread < self.lower_band[-1] and vol_ratio > 1
            short_signal = spread > self.upper_band[-1] and vol_ratio > 1

            if long_signal or short_signal:
                # 🪐 Calculate meteor size
                risk_amount = self.risk_per_trade * self.equity
                position_size = risk_amount