Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints: 🌙✨

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'openinterest': 'OpenInterest'
})

class LiquidationBreakout(Strategy):
    risk_pct = 0.02  # Max 2% risk per trade
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    
    def init(self):
        # Core indicators
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(talib.BBANDS, self.data.Close, 
                                                             timeperiod=self.bb_period, 
                                                             nbdevup=self.bb_dev, 
                                                             nbdevdn=self.bb_dev)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Derived metrics
        self.bb_width = self.I(lambda: (self.upper_bb - self.lower_bb) / self.middle_bb)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, 20)
        
        print("🌌 MOON DEV INDICATORS INITIALIZED 🌌 | BBANDS | ATR | Swing Levels | Volume MA")
        
    def next(self):
        if len(self.data) < 50:  # Warmup period
            return
            
        current_close = self.data.Close[-1]
        current_oi = self.data.OpenInterest[-1]
        prev_oi = self.data.OpenInterest[-2] if len(self.data.OpenInterest) > 1 else current_oi
        
        # Moon Dev Condition Checks 🌙
        oi_drop = (prev_oi - current_oi) / prev_oi >= 0.15
        bb_expanding = self.bb_width[-1] > self.bb_width_ma[-1]
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
        
        # Long Setup 🌕
        long_trigger = (oi_drop and
                       current_close > self.upper_bb[-1] and
                       current_close > self.swing_high[-2] and
                       self.data.Close[-2] < self.middle_bb[-2] and
                       bb_expanding and
                       volume_ok)
        
        # Short Setup 🌑
        short_trigger = (oi_drop and
                        current_close < self.lower_bb[-1] and
                        current_close < self.swing_low[-2] and
                        self.data.Close[-2] > self.middle_bb[-2] and
                        bb_expanding and
                        volume_ok)
        
        # Risk Management 🛡️
        if not self.position:
            atr_value = self.atr[-1]
            if long_trigger:
                sl = current_close - 1.5 * atr_value
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (current_close - sl)))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, 
                            tp=current_close + 3*atr_value)
                    print(f"🚀 MOON DEV LONG LAUNCH 🚀 | Size: {position_size} | Entry: {current_close} | SL: {sl:.2f} | TP: {current_close + 3*atr_value:.2f}")
            
            elif short_trigger:
                sl = current_close + 1.5 * atr_value
                risk_amount = self.equ