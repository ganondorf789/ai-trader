```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import pandas_ta

# Data preparation
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexFisher(Strategy):
    def init(self):
        # Calculate Vortex Indicator
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        # True Range
        tr = talib.TRANGE(high, low, close)
        
        # VM+ and VM-
        vm_plus = high - pd.Series(low).shift(1).values
        vm_plus[0] = 0
        vm_minus = pd.Series(high).shift(1).values - low
        vm_minus[0] = 0
        
        # Sum components
        sum_vm_plus = talib.SUM(vm_plus, 14)
        sum_vm_minus = talib.SUM(vm_minus, 14)
        sum_tr = talib.SUM(tr, 14)
        
        # VI calculations
        self.vi_plus = self.I(lambda: sum_vm_plus / sum_tr, name='VI+')
        self.vi_minus = self.I(lambda: sum_vm_minus / sum_tr, name='VI-')
        
        # Fisher Transform
        fisher = pandas_ta.fisher(high=high, low=low, length=10)
        self.fisher = self.I(lambda: fisher.iloc[:, 0], name='Fisher')
        
    def next(self):
        if len(self.data) < 15:
            return
        
        vi_plus = self.vi_plus
        vi_minus = self.vi_minus
        fisher = self.fisher
        
        # Moon Dev risk parameters 🌙
        risk_pct = 0.01
        swing_period = 5
        
        # Long entry logic
        if not self.position:
            if ((vi_plus[-2] < vi_minus[-2] and vi_plus[-1] > vi_minus[-1]) and \
               (fisher[-1] > -2 and fisher[-2] <= -2) and \
               (self.data.Volume[-1] > self.data.Volume[-2])):
                
                swing_low = talib.MIN(self.data.Low, swing_period)[-1]
                risk_amount = self.equity * risk_pct
                risk_per_share = self.data.Close[-1] - swing_low
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.buy(size=size, sl=swing_low)
                    print(f"🌙✨ BULLISH VORTEX DETECTED! Long {size} @ {self.data.Close[-1]} 🚀")

        # Short entry logic        
        if not self.position:
            if ((vi_minus[-2] < vi_plus[-2] and vi_minus[-1] > vi_plus[-1])) and \
               (fisher[-1] < 2 and fisher[-2] >= 2) and \
               (self.data.Volume[-1] > self.data.Volume[-2])):
                
                swing_high = talib.MAX(self.data.High, swing_period)[-1]
                risk_amount = self.equity * risk_pct
                risk_per_share = swing_high - self.data.Close[-1]
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.sell(size=size, sl=swing_high)
                    print(f"🌙✨ BEARISH VORTEX SPOTTED! Short {size} @ {self.data.Close[-1]} 🌪️")

        # Exit logic
        if self.position:
            if (self.position.is_long and (vi_minus[-2] < vi_plus[-2] and vi_minus[-1] > vi_plus[-1])) or \
               (self.position.is_short and (vi_plus[-2] < vi_minus[-2] and vi_plus[-1] > vi_minus[-1])) or \
               (abs(fisher[-1]) >