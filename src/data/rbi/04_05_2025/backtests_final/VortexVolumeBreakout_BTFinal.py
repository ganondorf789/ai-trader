Here's the complete fixed code with Moon Dev themed debug prints and proper technical implementation:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION - VORTEX VOLUME BREAKOUT STRATEGY 🌙
# 🚀 PACKAGE AI CERTIFIED - PROPER POSITION SIZING & RISK MANAGEMENT 🚀

# 1. IMPORT NECESSARY LIBRARIES
from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np

# 2. DATA PREPARATION
def load_data(path):
    # 🌙 LOAD AND CLEAN DATA
    data = pd.read_csv(path)
    
    # 🧹 CLEAN COLUMN NAMES
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # 🗺️ PROPER COLUMN MAPPING
    required_columns = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # 🛠️ COLUMN VALIDATION
    for original, new in required_columns.items():
        if original in data.columns:
            data.rename(columns={original: new}, inplace=True)
        else:
            raise ValueError(f"🌑 MOON DEV ERROR: Missing required column - {original}")
    
    # ⏰ SET DATETIME INDEX
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
    else:
        raise ValueError("🌑 MOON DEV ERROR: 'datetime' column not found")
    
    return data

# 3. STRATEGY IMPLEMENTATION
class VortexVolumeBreakout(Strategy):
    vortex_length = 14
    volume_window = 50
    swing_period = 20
    risk_percent = 0.01  # 1% risk per trade
    fib_levels = [0.382, 0.5, 0.618]
    
    def init(self):
        # 🌪️ CALCULATE VORTEX INDICATOR
        vortex = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vortex_length
        )
        self.vi_plus = self.I(lambda: vortex[f'VORTICSP_{self.vortex_length}'], name='VI+')
        self.vi_minus = self.I(lambda: vortex[f'VORTICSM_{self.vortex_length}'], name='VI-')
        
        # 📊 CALCULATE INDICATORS USING TA-LIB
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_window)
        self.resistance = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.support = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        # 💰 RISK MANAGEMENT CALCULATIONS
        current_equity = self.equity
        position_size = 0
        
        # 🚀 LONG ENTRY CONDITIONS
        if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and \
           (self.data.Volume[-1] > 1.5 * self.volume_avg[-1]) and \
           (self.data.Close[-1] > self.resistance[-1]):
            
            # 🛑 STOP LOSS CALCULATION
            sl_price = min(self.support[-1], self.data.Close[-1] - 1.5 * self.atr[-1])
            risk_per_share = self.data.Close[-1] - sl_price
            
            if risk_per_share > 0:
                # 🌕 PROPER POSITION SIZING (ROUNDED TO WHOLE SHARES)
                position_size = int(round((current_equity * self.risk_percent) / risk_per_share))
                fib_range = self.resistance[-1] - self.support[-1]
                
                # 🎯 FIBONACCI TARGETS
                targets = [self.data.Close[-1] + fib_range * lvl for lvl in self.fib_levels]
                
                # 🌕 MOON DEV ENTRY SIGN