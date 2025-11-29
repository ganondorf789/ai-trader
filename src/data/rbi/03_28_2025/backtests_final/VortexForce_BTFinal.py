I'll debug the code while maintaining the original strategy logic and adding Moon Dev themed improvements. Here's the fixed version:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and preprocess data 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexForce(Strategy):
    vi_period = 14
    efi_period = 13
    cmo_period = 20
    risk_pct = 0.01
    swing_period = 20
    
    def init(self):
        # Vortex Indicator 🌪️
        vi = ta.vortex(self.data.High, self.data.Low, self.data.Close, self.vi_period)
        self.vi_plus = self.I(lambda: vi[f'VTXP_{self.vi_period}'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VTXM_{self.vi_period}'], name='VI-')
        
        # Elder Force Index 💪
        mom = self.I(talib.MOM, self.data.Close, 1)
        force_index = mom * self.data.Volume
        self.efi = self.I(talib.EMA, force_index, self.efi_period)
        
        # Chande Momentum Oscillator 📈
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period)
        
        # Swing levels ⚖️
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        self.prev_low = self.prev_high = self.prev_efi = None

    def calculate_risk(self, direction, stop_price):
        """Moon Dev Risk Management Protocol 🌙"""
        risk_amount = self.equity * self.risk_pct
        if direction == 'long':
            price_diff = self.data.Close[-1] - stop_price
        else:  # short
            price_diff = stop_price - self.data.Close[-1]
            
        position_size = risk_amount / price_diff
        position_size = round(position_size)  # Ensure whole units
        
        print(f"🌙 Calculating Moon Position Size: {position_size} units")
        return position_size

    def next(self):
        if len(self.data) < self.swing_period + 2:
            return
        
        # Moon Dev signal tracking 🌙
        price_low = self.data.Low[-1]
        price_high = self.data.High[-1]
        efi_val = self.efi[-1]
        
        # Long entry constellation 🌟 (replaced crossover)
        if (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1]):
            if self.prev_low is not None and self.prev_efi is not None:
                if (price_low < self.prev_low) and (efi_val > self.prev_efi):
                    sl = self.swing_low[-1] * 0.995
                    size = self.calculate_risk('long', sl)
                    self.buy(size=size, sl=sl)
                    print(f"🌙 MOON DEV LONG SIGNAL ACTIVATED ✨ at {self.data.Close[-1]}")
        
        # Short entry constellation 🌑 (replaced crossover)
        elif (self.vi_minus[-2] > self.vi_plus[-2]) and (self.vi_minus[-1] < self.vi_plus[-1]):
            if self.prev_high is not None and self.prev_efi is not None:
                if (price_high > self.prev_high) and (efi_val < self.prev_efi):
                    sl = self.swing_high[-1] * 1.005
                    size = self.calculate_risk('short', sl)
                    self.sell(size=size, sl=sl)
                    print(f"🌙 MOON DEV SHORT SIGNAL ACTIV