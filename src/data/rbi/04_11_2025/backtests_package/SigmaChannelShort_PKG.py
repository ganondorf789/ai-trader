I've analyzed the code and it's already clean from any `backtesting.lib` imports or functions - great job! 🌙✨ The strategy uses proper indicator implementations with talib and custom calculations. Here's the verified code with some Moon Dev themed improvements:

```python
# 🌙 Moon Dev Backtest AI Generated Code - SigmaChannelShort Strategy
# ✅ Certified backtesting.lib FREE by Moon Dev Package AI 🌙

import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# --------------------------
# 🌌 DATA PREPROCESSING
# --------------------------
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# ✨ Clean column names with Moon Dev precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Map columns to backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'Funding_Rate'  # Must exist in original data
}, inplace=True)

# 🕒 Convert datetime with cosmic alignment
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# --------------------------
# 🚀 STRATEGY IMPLEMENTATION
# --------------------------
class SigmaChannelShort(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    trade_duration = 192  # 48 hours in 15m intervals
    
    def init(self):
        # 🌗 Funding Rate Indicators
        fr = self.data.Funding_Rate.fillna(method='ffill')
        self.funding_30d_avg = self.I(talib.SMA, fr, 2880, name='FR_30D_AVG')
        self.funding_30d_std = self.I(talib.STDDEV, fr, 2880, name='FR_30D_STD')
        
        # ✨ Keltner Channel Indicators (Pure talib implementation)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr14 = self.I(talib.ATR, 
                          self.data.High, 
                          self.data.Low, 
                          self.data.Close, 
                          timeperiod=14, 
                          name='ATR14')
        
        # 🌑 Custom band calculations without backtesting.lib
        self.lower_band = self.I(lambda: self.ema20 - 2*self.atr14, name='LOWER_BAND')
        self.upper_band = self.I(lambda: self.ema20 + 2*self.atr14, name='UPPER_BAND')
        
        self.trade_entry_bar = None

    def next(self):
        # 🌙 Moon Dev Core Trade Logic
        if not self.position:
            # Entry Conditions (Pure numpy/pandas conditions)
            fr = self.data.Funding_Rate[-1]
            fr_avg = self.funding_30d_avg[-1]
            fr_std = self.funding_30d_std[-1]
            
            cond1 = fr > (fr_avg + 3*fr_std) if all([fr, fr_avg, fr_std]) else False
            cond2 = self.data.Close[-1] < self.lower_band[-1]
            
            if cond1 and cond2:
                # 🌗 Cosmic Risk Management
                entry_price = self.data.Close[-1]
                sl_price = self.upper_band[-1]
                risk_per_unit = sl_price - entry_price
                
                if risk_per_unit <= 0:
                    print("🌑 MOON DEV ERROR: Negative risk detected! Aborting launch sequence.")
                    return
                
                position_size = int(round((self.equity * self.risk_percent) / risk_per_unit))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl_price, tag='SHORT_ENTRY')
                    self.trade_entry_bar = len(self.data)-1
                    print(f"🚀🌙 MOON DEV SHORT SIGNAL! Size: {position_size} @ {entry_price}")

        else:
            # Exit Conditions (Time and indicator based)
            current_bar = len(self.data)-1
            bars_held = current_bar - self.trade_entry_bar
            
            # ⏳ Time-based Exit (No backtesting.lib used)