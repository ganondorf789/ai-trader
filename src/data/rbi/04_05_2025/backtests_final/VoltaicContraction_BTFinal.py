Here's the fully debugged and fixed version of your VoltaicContraction strategy with Moon Dev themed debug prints and proper backtesting implementation:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaicContraction(Strategy):
    risk_percent = 0.02  # 2% risk per trade as per strategy rules
    position_size = None
    entry_bb_width = None
    entry_price = None
    
    def init(self):
        # Core indicators - all using talib 🌙
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(talib.BBANDS, 
                                                             self.data.Close, 
                                                             timeperiod=20, 
                                                             nbdevup=2, 
                                                             nbdevdn=2, 
                                                             matype=0)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=0.02, maximum=0.2)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        self.pending_long = []
        self.pending_short = []

    def next(self):
        current_bar = len(self.data) - 1
        print(f"🌙 Moon Dev Processing Bar {current_bar}...")
        
        # Check pending entries
        self.check_pending_entries()
        
        # Entry logic
        if not self.position:
            self.check_long_entries()
            self.check_short_entries()
        
        # Exit logic
        self.check_exits()

    def check_pending_entries(self):
        for entry in self.pending_long[:]:
            bars_since = len(self.data) - 1 - entry['bar']
            if bars_since > 3:
                self.pending_long.remove(entry)
                print(f"🌑 Moon Dev: Long entry expired")
                continue
                
            if self.adx[-1] > 30:
                self.execute_long(entry)
                self.pending_long.remove(entry)

    def check_long_entries(self):
        if len(self.data) < 40:
            return
            
        # Consolidation checks
        consolidation = all(
            (self.data.High[i] <= self.upper_bb[i] and 
            self.data.Low[i] >= self.lower_bb[i] and
            self.upper_bb[i] <= (self.ema20[i] + 2.5*self.atr20[i]) and
            self.lower_bb[i] >= (self.ema20[i] - 2.5*self.atr20[i]) and
            (self.upper_bb[i] - self.lower_bb[i]) < 0.5*(self.ema20[i] + 2.5*self.atr20[i] - (self.ema20[i] - 2.5*self.atr20[i]))
            for i in range(-6, -1)
        )
        
        if not consolidation:
            return
            
        # Breakout check
        if (self.data.Close[-2] <= self.upper_bb[-2] and 
            self.data.Close[-1] > self.upper_bb[-1] and 
            self.data.Volume[-1] > self.volume_sma[-2]):
            
            print(f"🚀 Moon Dev: Long breakout detected!")
            self.pending_long.append({
                'bar': len(self.data) - 1,
                'upper_bb': self.upper_bb[-1],
                'lower_bb': self.lower_bb[-1],
                'atr': self.atr14[-1]
            })

    def execute_long(self, entry):
        # Calculate position size based on 2% risk
        stop_loss = entry['lower_bb']
        risk_amount = self.equity * self.risk_percent
        risk_per_share = self.data.Close[-1] - stop_loss
        position_size = risk_amount / risk_per_share
        
        # Convert to percentage of equity
        position_size = min(position_size / self.equity, 0.5)  # Cap at