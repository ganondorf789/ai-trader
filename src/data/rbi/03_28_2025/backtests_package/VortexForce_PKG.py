Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

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

    def next(self):
        if len(self.data) < self.swing_period + 2:
            return
        
        # Moon Dev signal tracking 🌙
        price_low = self.data.Low[-1]
        price_high = self.data.High[-1]
        efi_val = self.efi[-1]
        
        # Long entry constellation 🌟 (replaced crossover)
        if (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1]):
            if self.prev_low and self.prev_efi:
                if (price_low < self.prev_low) and (efi_val > self.prev_efi):
                    sl = self.swing_low[-1] * 0.995
                    self.calculate_risk('long', sl)
                    print(f"🌙 MOON DEV LONG SIGNAL ACTIVATED ✨ at {self.data.Close[-1]}")
        
        # Short entry constellation 🌑 (replaced crossover)
        elif (self.vi_minus[-2] > self.vi_plus[-2]) and (self.vi_minus[-1] < self.vi_plus[-1]):
            if self.prev_high and self.prev_efi:
                if (price_high > self.prev_high) and (efi_val < self.prev_efi):
                    sl = self.swing_high[-1] * 1.005
                    self.calculate_risk('short', sl)
                    print(f"🌙 MOON DEV SHORT SIGNAL ACTIVATED ✨ at {self.data.Close[-1]}")
        
        # Moon exit protocol 🚪
        if self.position:
            if self.position.is_long and self.cmo[-1] < 70 and self.cmo[-2] >= 70:
                self.position.close()
                print(f"🌙 MOON DEV LONG EXIT ✨ at {self.data.Close[-1]}")
            elif self.position.is_short and self.cmo[-1] > 30 and self.cmo[-2] <= 30:
                self.position.close()
                print(f"🌙 MOON DEV SHORT EXIT ✨ at {self.data.Close[-1]}")
        
        # Update previous values 🌗
        self.prev_low = price_low
        self.prev_high = price_high
        self.prev