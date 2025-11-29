To address the problem, here's a complete implementation of the VortexSqueeze strategy in backtesting.py:

```python
# moon_dev_vortex_squeeze.py
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

class VortexSqueeze(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Calculate Vortex Indicator using TA-Lib
        vi_plus, vi_minus = talib.VORTEX(self.data.High, self.data.Low, self.data.Close, 14)
        self.I(lambda: vi_plus, name='VI_PLUS')
        self.I(lambda: vi_minus, name='VI_MINUS')
        
        # Calculate Bollinger Bands with TA-Lib
        upper, middle, lower = talib.BBANDS(self.data.Close, 20, 2, 2, 0)
        self.I(lambda: upper, name='BB_UPPER')
        self.I(lambda: middle, name='BB_MIDDLE')
        self.I(lambda: lower, name='BB_LOWER')
        
        # Calculate Bollinger Bandwidth
        bandwidth = (upper - lower) / middle
        self.I(lambda: bandwidth, name='BB_WIDTH')
        
        # Calculate 20-period average of bandwidth
        bb_width_avg = talib.SMA(bandwidth, 20)
        self.I(lambda: bb_width_avg, name='BB_WIDTH_AVG')
        
    def next(self):
        # Moon Dev Safety Checks 🌙
        if len(self.data) < 20 or np.isnan(self.data.BB_WIDTH_AVG[-1]):
            return

        # Current Values ✨
        vi_plus = self.data.VI_PLUS[-1]
        vi_minus = self.data.VI_MINUS[-1]
        close = self.data.Close[-1]
        upper_bb = self.data.BB_UPPER[-1]
        lower_bb = self.data.BB_LOWER[-1]
        bb_width = self.data.BB_WIDTH[-1]
        bb_width_avg = self.data.BB_WIDTH_AVG[-1]

        # Moon Dev Trading Logic 🌙🚀
        if not self.position:
            # Long Entry Constellation ✨
            if (crossover(self.data.VI_PLUS, self.data.VI_MINUS) and
                bb_width < bb_width_avg and
                close > upper_bb):
                
                risk_amount = self.equity * self.risk_percent
                stop_loss = lower_bb
                risk_per_share = close - stop_loss
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    tp = close + 2 * risk_per_share
                    self.buy(size=size, sl=stop_loss, tp=tp)
                    print(f"🌙🚀 LONG SIGNAL! Entry: {close:.2f} | Size: {size} | VI+ {vi_plus:.2f} > VI- {vi_minus:.2f} | BB Width {bb_width:.4f} < Avg {bb_width_avg:.4f}")

            # Short Entry Constellation 🌑
            elif (crossover(self.data.VI_MINUS, self.data.VI_PLUS) and
                  bb_width < bb_width_avg and
                  close < lower_bb):
                  
                risk_amount = self.equity * self.risk_percent
                stop_loss = upper_bb
                risk_per_share = stop_loss - close
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    tp = close - 2 * risk_per_share
                    self.sell(size=size, sl=stop_loss, tp=tp)
                    print(f"🌑🌙 SHORT SIGNAL! Entry: {close:.2f} | Size: {size} | VI- {vi_minus:.2f} > VI+ {vi_plus:.2f} | BB Width {bb_width:.4f} < Avg {bb_width_avg:.4f}")

# Moon Dev Data Preparation 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    '