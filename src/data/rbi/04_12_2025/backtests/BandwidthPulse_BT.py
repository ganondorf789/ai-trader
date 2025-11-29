```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

class BandwidthPulse(Strategy):
    risk_pct = 0.01
    max_bars = 5 * 24 * 4  # 5 days in 15m intervals
    
    def init(self):
        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(ta.bbands, self.data.Close, length=20, std=2, name=['Upper', 'Middle', 'Lower'])
        
        # Bandwidth calculations
        self.bandwidth = self.I(lambda: (self.upper - self.lower) / self.middle, name='Bandwidth')
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, 20, name='BW_Min')
        
        # Volume indicators
        self.volume_roc = self.I(talib.ROC, self.data.Volume, 14, name='VROC')
        
        # Volatility filter
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR')
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SwingLow')
        
        self.contraction_range = None

    def next(self):
        print(f"🌙 Processing {self.data.index[-1]}... Price: {self.data.Close[-1]:.2f} ✨")
        
        if len(self.data) < 20 or not self.contraction_range:
            return

        current_close = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        atr = self.atr[-1]

        # Long entry logic
        if current_close > upper and (current_close - upper) > atr:
            sl = max(self.swing_low[-1], lower)
            risk = current_close - sl
            if risk <= 0: return
            
            size = int(round((self.equity * self.risk_pct) / risk))
            target1 = current_close + self.contraction_range * 1.618
            target2 = current_close + self.contraction_range * 2.618
            
            self.buy(size=size, sl=sl, tp=target1, tag='Long')
            print(f"🚀 LONG ENTRY: {current_close:.2f} | Size: {size} | Risk: {risk:.2f} 🌕")
            self.contraction_range = None  # Reset after entry

        # Short entry logic
        elif current_close < lower and (lower - current_close) > atr:
            sl = min(self.swing_high[-1], upper)
            risk = sl - current_close
            if risk <= 0: return
            
            size = int(round((self.equity * self.risk_pct) / risk))
            target1 = current_close - self.contraction_range * 1.618
            target2 = current_close - self.contraction_range * 2.618
            
            self.sell(size=size, sl=sl, tp=target1, tag='Short')
            print(f"🌑 SHORT ENTRY: {current_close:.2f} | Size: {size} | Risk: {risk:.2f} 🌘")
            self.contraction_range = None

        # Update contraction range
        if self.bandwidth[-1] <= self.bandwidth_min[-1] and self.volume_roc[-1] > self.volume_roc[-2]:
            self.contraction_range = self.upper[-1] - self.lower[-1]
            print(f"🔔 BANDWIDTH PULSE! Range