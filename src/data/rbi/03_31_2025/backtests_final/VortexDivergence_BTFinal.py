Here's the fixed backtest code with all technical issues resolved while preserving the original strategy logic:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexDivergence(Strategy):
    vi_period = 14
    efi_period = 13
    kc_period = 20
    kc_multiplier = 2
    volume_filter = 20
    risk_pct = 0.01
    atr_period = 20
    cooldown = 5
    consecutive_losses = 0

    def init(self):
        # Calculate indicators using proper self.I() wrapper
        h, l, c, v = self.data.High, self.data.Low, self.data.Close, self.data.Volume
        
        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(ta.vortex,
                                             h, l, c,
                                             length=self.vi_period,
                                             name=['VI+', 'VI-'])
        
        # Elder's Force Index
        force = (c - c.shift(1)) * v
        self.efi = self.I(talib.EMA, force, self.efi_period, name='EFI')
        
        # Keltner Channels
        self.ema = self.I(talib.EMA, c, self.kc_period, name='EMA')
        atr = self.I(talib.ATR, h, l, c, self.kc_period, name='ATR')
        self.upper_kc = self.I(lambda: self.ema + atr*self.kc_multiplier,
                              name='Upper KC')
        self.lower_kc = self.I(lambda: self.ema - atr*self.kc_multiplier,
                              name='Lower KC')
        
        # Volume filter
        self.vol_ma = self.I(talib.SMA, v, self.volume_filter, name='Vol MA')
        
        # Swing detection
        self.swing_high = self.I(talib.MAX, h, 5, name='Swing High')
        self.swing_low = self.I(talib.MIN, l, 5, name='Swing Low')
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, h, l, c, self.atr_period, name='ATR')

    def next(self):
        # Moon Dev themed debug prints 🌙✨
        if len(self.data) % 100 == 0:
            print(f"\n🌙 Moon Dev Status Update @ {self.data.index[-1]} ✨")
            print(f"Equity: ${self.equity:,.2f} | Price: ${self.data.Close[-1]:.2f}")
        
        # Cooldown check
        if self.consecutive_losses >= 3:
            print("🚨🌙 3 consecutive losses! Cooling down... ⏳")
            return

        if not self.position:
            # Volume filter
            if self.data.Volume[-1] < self.vol_ma[-1]:
                return

            # Long entry logic
            if (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1]):
                bull_div = (self.data.Low[-1] < self.data.Low[-2]) and \
                          (self.efi[-1] > self.efi[-2])
                if bull_div:
                    sl = self.swing_low[-1]
                    risk = self.data.Close[-1] - sl
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        self.buy(size=size, sl=sl, tp=self.upper_kc[-1])
                        print(f"🚀🌙 LONG Entry @ {self.data.Close[-1]:.2f}")
                        print(f"