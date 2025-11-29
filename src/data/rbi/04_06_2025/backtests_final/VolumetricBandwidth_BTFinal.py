I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VolumetricBandwidth(Strategy):
    vwma_period = 20
    bb_period = 20
    bb_dev = 2
    median_period = 20
    sma_period = 50
    bandwidth_threshold = 1.5
    exit_threshold = 0.1
    max_bars_held = 10
    risk_pct = 0.01

    def init(self):
        # Calculate indicators with self.I()
        # VWMA = (SMA(Close*Volume) / SMA(Volume))
        self.vwma = self.I(lambda c, v, p: talib.SMA(c*v, p) / talib.SMA(v, p),
                          self.data.Close, self.data.Volume, self.vwma_period)
        
        # Bollinger Bands Width
        def bb_width(close):
            upper, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return upper - lower
        self.bb_width = self.I(bb_width, self.data.Close)
        
        # Median and SMA of BB Width
        self.bb_median = self.I(talib.MEDIAN, self.bb_width, self.median_period)
        self.bb_sma50 = self.I(talib.SMA, self.bb_median, self.sma_period)
        
        # Swing levels for stops
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        
        self.entry_bar = 0

    def next(self):
        # Skip early bars
        if len(self.data) < max([self.sma_period, self.bb_period, 20]) + 1:
            return
        
        current_close = self.data.Close[-1]
        vwma_val = self.vwma[-1]
        bb_width_val = self.bb_width[-1]
        bb_med_val = self.bb_median[-1]
        bb_sma50_val = self.bb_sma50[-1]
        
        # 🌙 Volatility filter
        if bb_med_val <= bb_sma50_val:
            print(f"🌙✨ Volatility too low! Median {bb_med_val:.2f} ≤ SMA50 {bb_sma50_val:.2f}")
            return
        
        if not self.position:
            # Long entry constellation 🌙🚀
            if current_close > vwma_val and bb_width_val > self.bandwidth_threshold * bb_med_val:
                sl_price = self.swing_low[-1]
                risk_amount = self.risk_pct * self.equity
                size = int(round(risk_amount / (current_close - sl_price)))
                
                if size > 0:
                    self.buy(size=size, sl=sl_price)
                    self.entry_bar = len(self.data)
                    print(f"🌙🚀 LONG LAUNCH! {current_close:.2f}")
                    print(f"    Size: {size}, SL: {sl_price:.2f}, Risk: {risk_amount:.2f}")
            
            # Short entry constellation 🌙🌑
            elif current_close < vwma_val and bb_width_val > self.bandwidth_threshold * bb_med_val:
                sl_price = self.swing_high[-1]
                risk_amount = self.risk_pct * self.equity
                size = int(round(risk_amount / (sl_price - current_close)))
                
                if size > 0:
                    self.sell(size=size, sl=sl_price)
                    self.entry_bar = len(self.data)
                    print(f"🌙🌑 SHORT LA