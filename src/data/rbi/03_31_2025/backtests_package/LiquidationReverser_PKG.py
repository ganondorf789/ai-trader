Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION ✨

import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# =====================
# DATA PREPARATION 🌐
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Standardize column names
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# =====================
# STRATEGY CORE 🚀
# =====================
class LiquidationReverser(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 20      # Swing high/low detection
    bb_period = 20         # Bollinger Bands period
    volatility_lookback = 480  # 5 days in 15m intervals (4*24*5)
    
    def init(self):
        # =====================
        # INDICATORS CALCULATION 🌗
        # =====================
        # Bollinger Bands components
        self.upper_band = self.I(talib.BBANDS, self.data.Close, 
                               timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.middle_band = self.I(talib.BBANDS, self.data.Close,
                                timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.lower_band = self.I(talib.BBANDS, self.data.Close,
                               timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # Bollinger Band Width
        self.bb_width = self.I(lambda close: ((self.upper_band - self.lower_band) / self.middle_band),
                             self.data.Close, name='BB Width')
        
        # 5-day low of BB Width
        self.bb_width_low = self.I(talib.MIN, self.bb_width, self.volatility_lookback)
        
        # Liquidation zones (swing levels)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing Low')
        
        self.entry_condition_met = False
        self.entry_bb_width = None

    def next(self):
        # =====================
        # MOON DEV CORE LOGIC 🌙
        # =====================
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Skip if not enough data
        if len(self.data) < self.swing_period * 2:
            return
            
        # =====================
        # ENTRY CONDITIONS CHECK 🔍
        # =====================
        if not self.position:
            # LONG ENTRY: Price at swing low + volatility contraction
            if (current_low <= self.swing_low[-1] and 
                self.bb_width[-1] <= self.bb_width_low[-1]):
                
                sl = self.swing_low[-1] * 0.999  # Stop below swing low
                risk_amount = self.risk_per_trade * self.equity
                risk_per_unit = current_close - sl
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    tp = self.swing_high[-1]  # Opposite cluster
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl, tp=tp)
                        self.entry_bb_width = self.bb_width[-1]
                        print(f"🌙 MOON DEV LONG ENTRY 🌟 | Price: {current_close:.2f} | "
                              f"Size: {position_size} | Risk: