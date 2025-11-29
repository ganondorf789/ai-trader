```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VortexSqueezeStrategy(Strategy):
    vi_period = 14
    bb_period = 20
    bb_std = 2
    bb_lookback = 100
    exit_bars = 15
    risk_pct = 0.02
    swing_period = 10

    def init(self):
        # Vortex Indicator calculations
        # Calculate VM+ and VM- using shifted values
        vm_plus = (self.data.High - self.data.Low.shift(1)).abs().fillna(0)
        vm_minus = (self.data.High.shift(1) - self.data.Low).abs().fillna(0)
        
        sum_vm_plus = self.I(talib.SUM, vm_plus, timeperiod=self.vi_period, name='SUM_VM+')
        sum_vm_minus = self.I(talib.SUM, vm_minus, timeperiod=self.vi_period, name='SUM_VM-')
        
        tr = self.I(talib.TRANGE, self.data.High, self.data.Low, self.data.Close, name='TR')
        sum_tr = self.I(talib.SUM, tr, timeperiod=self.vi_period, name='SUM_TR')
        
        self.vi_plus = self.I(lambda a, b: a/b, sum_vm_plus, sum_tr, name='VI+')
        self.vi_minus = self.I(lambda a, b: a/b, sum_vm_minus, sum_tr, name='VI-')

        # Bollinger Band Width calculation
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, self.bb_std, self.bb_std)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB_WIDTH')

        self.entry_bar = 0

    def next(self):
        # Moon Dev debug prints
        print(f"🌙 Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} | VI+={self.vi_plus[-1]:.4f}, VI-={self.vi_minus[-1]:.4f} | BBW={self.bb_width[-1]:.4f}")

        if not self.position:
            # Entry logic
            if crossover(self.vi_plus, self.vi_minus):
                bb_values = self.bb_width[-self.bb_lookback:] if len(self.bb_width) >= self.bb_lookback else self.bb_width
                bb_threshold_low = np.percentile(bb_values, 20)
                
                if self.bb_width[-1] < bb_threshold_low:
                    # Risk management calculations
                    entry_price = self.data.Close[-1]
                    swing_low = np.min(self.data.Low[-self.swing_period:])
                    stop_loss = min(swing_low, entry_price * 0.98)
                    
                    risk_amount = self.equity * self.risk_pct
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share <= 0:
                        print("🚨 Moon Dev Alert: Invalid risk per share")
                        return
                    
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        self.entry_bar = len(self.data) - 1
                        print(f"🚀🌙 MOON DEV LONG ENTRY 🚀 | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f}")
        else:
            # Exit logic
            exit_conditions = []
            
            # BB Width exit
            bb_values = self.bb_width[-self.bb_lookback:] if len(self.bb_width) >= self.bb