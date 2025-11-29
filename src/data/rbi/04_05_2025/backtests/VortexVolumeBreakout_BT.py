```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION - VORTEX VOLUME BREAKOUT STRATEGY 🌙

# 1. IMPORT NECESSARY LIBRARIES
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib
import os

# 2. DATA PREPARATION
def load_data(path):
    # 🌙 LOAD AND CLEAN DATA
    data = pd.read_csv(path)
    
    # 🧹 CLEAN COLUMN NAMES
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # 🗺️ PROPER COLUMN MAPPING
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # ⏰ SET DATETIME INDEX
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# 3. STRATEGY IMPLEMENTATION
class VortexVolumeBreakout(Strategy):
    vortex_length = 14
    volume_window = 50
    swing_period = 20
    risk_percent = 0.01
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
        equity = self.equity
        position_size = 0
        
        # 🚀 LONG ENTRY CONDITIONS
        if (crossover(self.vi_plus, self.vi_minus) and
            self.data.Volume[-1] > 1.5 * self.volume_avg[-1] and
            self.data.Close[-1] > self.resistance[-1]):
            
            # 🛑 STOP LOSS CALCULATION
            sl_price = min(self.support[-1], self.data.Close[-1] - 1.5*self.atr[-1])
            risk_per_share = self.data.Close[-1] - sl_price
            
            if risk_per_share > 0:
                position_size = int(round((equity * self.risk_percent) / risk_per_share))
                fib_range = self.resistance[-1] - self.support[-1]
                
                # 🎯 FIBONACCI TARGETS
                targets = [self.data.Close[-1] + fib_range * lvl for lvl in self.fib_levels]
                
                # 🌕 MOON DEV ENTRY SIGNAL
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=sl_price,
                        tag={'targets': targets, 'entry_price': self.data.Close[-1]}
                    )
                    print(f"🌕 MOON DEV LONG SIGNAL 🌕 | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl_price:.2f}")
        
        # 🪂 SHORT ENTRY CONDITIONS
        elif (crossover(self.vi_minus, self.vi_plus) and
              self.data.Volume[-1] > 1.5 * self.volume_avg[-1] and
              self.data.Close[-1] < self.support[-1]):
            
            # 🛑 STOP LOSS CALCULATION
            sl_price = max(self.resistance[-1], self.data.Close[-1