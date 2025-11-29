```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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
        
        # Long entry constellation 🌟
        if crossover(self.vi_plus, self.vi_minus):
            if self.prev_low and self.prev_efi:
                if (price_low < self.prev_low) and (efi_val > self.prev_efi):
                    sl = self.swing_low[-1] * 0.995
                    self.calculate_risk('long', sl)
        
        # Short entry constellation 🌑
        elif crossover(self.vi_minus, self.vi_plus):
            if self.prev_high and self.prev_efi:
                if (price_high > self.prev_high) and (efi_val < self.prev_efi):
                    sl = self.swing_high[-1] * 1.005
                    self.calculate_risk('short', sl)
        
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
        self.prev_efi = efi_val

    def calculate_risk(self, trade_type, sl_price):
        equity = self.equity
        close = self.data.Close[-1]
        risk_amount = equity * self.risk_pct
        
        if trade_type == 'long':
            risk_per_unit = close - sl_price
            tp_price = close + (risk_per_unit * 2)
        else:
            risk_per_unit = sl_price - close
            tp_price = close - (risk_per_unit * 2)
            
        if risk_per_unit <=