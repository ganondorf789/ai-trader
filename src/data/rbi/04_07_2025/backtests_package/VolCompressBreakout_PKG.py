Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev Backtest AI Implementation
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data Preparation 🌍
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)
data['Date'] = pd.to_datetime(data['date'])
data.set_index('Date', inplace=True)

class VolCompressBreakout(Strategy):
    # Strategy Parameters 🎛️
    bb_period = 20
    bb_dev = 2
    bbw_percentile_window = 200
    risk_pct = 0.01
    sar_accel = 0.02
    sar_max = 0.2
    atr_period = 14
    volume_ratio_window = 5
    
    def init(self):
        # Bollinger Bands & Bandwidth 🌗
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                    timeperiod=self.bb_period, 
                                                    nbdevup=self.bb_dev,
                                                    nbdevdn=self.bb_dev)
        
        # Calculate BBW Percentile without backtesting.lib
        def bbw_percentile_calc(bbw):
            return bbw.rolling(self.bbw_percentile_window).apply(
                lambda y: (y <= y[-1]).mean() * 100)
            
        bbw = (self.upper - self.lower) / self.middle
        self.bbw_percentile = self.I(bbw_percentile_calc, bbw)
        
        # Volume Ratio Calculation 🌊
        close_diff = self.I(lambda x: x.diff(), self.data.Close)
        up_vol = self.I(lambda x: np.where(close_diff > 0, self.data.Volume, 0), self.data.Close)
        down_vol = self.I(lambda x: np.where(close_diff < 0, self.data.Volume, 0), self.data.Close)
        up_sum = self.I(lambda x: x.rolling(self.volume_ratio_window).sum(), up_vol)
        down_sum = self.I(lambda x: x.rolling(self.volume_ratio_window).sum(), down_vol)
        self.vol_ratio = up_sum / (down_sum + 1e-9)
        
        # Trend Following Indicators 🚦
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                         acceleration=self.sar_accel, maximum=self.sar_max)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                         timeperiod=self.atr_period)
        
    def next(self):
        # Moon Dev Safety Checks 🌙
        if len(self.data.Close) < self.bbw_percentile_window:
            return
        
        current_bbw_pct = self.bbw_percentile[-1]
        prev_bbw_pct = self.bbw_percentile[-2] if len(self.bbw_percentile) > 1 else 100
        
        # Entry Signal Detection 🚨
        if not self.position:
            # Volatility Compression Breakout Pattern Detected! 🔭
            if current_bbw_pct < 20 and prev_bbw_pct >= 20:
                # Long Entry Conditions 🚀
                if self.vol_ratio[-1] > 2.0 and self.data.Close[-1] > self.upper[-1]:
                    entry_price = self.data.Close[-1]
                    atr_val = self.atr[-1]
                    sl_price = max(self.lower[-1], entry_price - 1.5*atr_val)
                    risk_amount = self.equity * self.risk_pct
                    risk_per_trade = entry_price - sl_price
                    
                    if risk_per_trade > 0:
                        size = int(round(risk_amount / risk_per_trade))
                        self.buy(size=size, sl=sl_price, 
                                tag="Moon Dev Long Entry 🌕")
                        print(f"