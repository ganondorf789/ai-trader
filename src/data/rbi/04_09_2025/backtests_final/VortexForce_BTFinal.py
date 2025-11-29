I'll fix the incomplete code while maintaining the original strategy logic and adding proper Moon Dev themed debug prints. Here's the complete fixed version:

```python
# 🌙 Moon Dev Backtest AI Generated Code
from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np

# =====================
# DATA PREPARATION
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format columns 🌙
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
data.set_index('datetime', inplace=True)

# =====================
# VORTEXFORCE STRATEGY
# =====================
class VortexForce(Strategy):
    vi_period = 14
    efi_period = 13
    entry_threshold = 0.5
    exit_threshold = 0.2
    risk_pct = 0.01  # 1% risk per trade
    swing_period = 20  # Swing high/low lookback
    
    def init(self):
        # 🌙 Vortex Indicator Calculation
        vi = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi[f'VORTIC_{self.vi_period}'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VORTIS_{self.vi_period}'], name='VI-')
        
        # 🌀 Elder Force Index Calculation
        self.efi = self.I(ta.efi, 
                         self.data.Close, 
                         self.data.Volume, 
                         length=self.efi_period,
                         name='EFI')
        
        # 🛑 Swing High/Low for Stops
        self.swing_low = self.I(talib.MIN, 
                              self.data.Low, 
                              timeperiod=self.swing_period,
                              name='Swing Low')
        self.swing_high = self.I(talib.MAX,
                               self.data.High,
                               timeperiod=self.swing_period,
                               name='Swing High')

    def next(self):
        current_price = self.data.Close[-1]
        print(f"🌙 Moon Dev Pulse: Price={current_price:.2f} | VI+={self.vi_plus[-1]:.2f} | VI-={self.vi_minus[-1]:.2f} | EFI={self.efi[-1]:.2f}")
        
        if not self.position:
            # 🚀 Long Entry Logic
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
                self.efi[-1] > self.entry_threshold):
                
                entry_price = current_price
                sl_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_pct
                stop_distance = entry_price - sl_price
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🚀🌕 BLASTOFF LONG! Size: {position_size} | Entry: {entry_price:.2f} | Cosmic SL: {sl_price:.2f}")
            
            # 🌑 Short Entry Logic
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and
                  self.efi[-1] < -self.entry_threshold):
                
                entry_price = current_price
                sl_price = self.swing_high[-1]
                risk_amount = self.equity * self.risk_pct
                stop_distance = sl_price - entry_price
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    self.sell(size=position_size, sl=sl_price)
                    print(f"🌒🌑 BLACKHOLE SHORT! Size: {position_size