```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION CLUSTER BREAKOUT STRATEGY 🚀

import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# =====================
# DATA PREPARATION 🌌
# =====================
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names 🧼
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
# STRATEGY CLASS 🌙✨
# =====================
class LiquidationClusterBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    bb_period = 20
    bb_dev = 2
    vwap_anchor = 'D'  # Daily anchor for VWAP
    momentum_fast = 12
    momentum_slow = 26
    
    def init(self):
        # =====================
        # INDICATORS 📊
        # =====================
        # VWAP Calculation 🌗
        def compute_vwap(h, l, c, v):
            return ta.vwap(high=h, low=l, close=c, volume=v, anchor=self.vwap_anchor)
        self.vwap = self.I(compute_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        
        # Bollinger Bands 🌈
        def bb_bands(close):
            return talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev, 0)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(bb_bands, self.data.Close)
        
        # Bollinger Band Width 🌐
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, 
                              self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, self.bb_period)
        self.bb_width_std = self.I(talib.STDDEV, self.bb_width, self.bb_period)
        
        # Volume Momentum Oscillator 🌀
        def momentum_oscillator(c, v):
            pv = c * v
            return talib.EMA(pv, self.momentum_fast) - talib.EMA(pv, self.momentum_slow)
        self.momentum = self.I(momentum_oscillator, self.data.Close, self.data.Volume)
        
        # Volume Analysis 📈
        self.vol_avg = self.I(talib.SMA, self.data.Volume, 20)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        self.peak_equity = self._equity_curve[-1]  # Track maximum equity 🌕

    def next(self):
        # =====================
        # RISK MANAGEMENT 🛡️
        # =====================
        # Check maximum drawdown 💥
        current_equity = self._equity_curve[-1]
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= 0.05:
            self.position.close()
            print(f"🌙 MAX DAILY LOSS HIT: {drawdown*100:.1f}% drawdown! Closing all positions 💣")
            return
        
        if self.position:
            self.manage_position()
        else:
            self.check_entries()

    def check_entries(self):
        # =====================
        # ENTRY CONDITIONS 🚪
        # =====================
        # Volatility contraction check 🌪️
        if self.bb_width[-1] >= self.bb_width_avg[-1]:
            return
        
        # Volume confirmation 🔊
        if self.data.Volume[-1] < 1.5 * self