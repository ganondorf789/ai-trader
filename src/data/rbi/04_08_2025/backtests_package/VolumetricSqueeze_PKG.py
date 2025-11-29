Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolumetricSqueeze Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# INDICATOR FUNCTIONS
# =====================
def calculate_vw_roc(close, volume, timeperiod):
    """🌊 Volume-Weighted Rate of Change (Moon Power Edition)"""
    roc = talib.ROC(close, timeperiod=timeperiod)
    vol_sum = talib.SUM(volume, timeperiod=timeperiod)
    return roc * vol_sum

# =====================
# STRATEGY CLASS
# =====================
class VolumetricSqueezeStrategy(Strategy):
    size = 1_000_000  # 🌕 Initial cosmic fuel
    
    def init(self):
        # =====================
        # CORE INDICATORS 🌌
        # =====================
        # Volume-Weighted ROC (3-day cosmic cycle)
        self.vw_roc = self.I(calculate_vw_roc, 
                           self.data.Close, 
                           self.data.Volume, 
                           288)  # 288 periods = 3 days in 15m
        
        # Bollinger Bands (Stellar Navigation System)
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        self.upper_band = self.middle_band + 2 * self.std_dev
        self.lower_band = self.middle_band - 2 * self.std_dev
        
        # Keltner Channel (Warp Drive Squeeze Detector)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_keltner = self.I(talib.ATR, 
                                self.data.High, 
                                self.data.Low, 
                                self.data.Close, timeperiod=14)
        self.keltner_upper = self.ema20 + 1.5 * self.atr_keltner
        self.keltner_lower = self.ema20 - 1.5 * self.atr_keltner
        
        # ATR Trailing Stop (Cosmic Shield)
        self.atr = self.I(talib.ATR, 
                        self.data.High, 
                        self.data.Low, 
                        self.data.Close, timeperiod=14)
        
        self.peak = 0  # 🌑 Track stellar price peaks

    def next(self):
        # =====================
        # MOON DEV DEBUG SYSTEM 🌙
        # =====================
        if len(self.data) % 100 == 0:
            print((f"🌙 Moon Pulse Check | Bar {len(self.data)} "
                   f"| Price: {self.data.Close[-1]:.2f} "
                   f"| VW ROC: {self.vw_roc[-1]:.2f} "
                   f"| ATR Shield: {self.atr[-1]:.2f} "
                   f"| Cosmic Fuel: {self.equity:,.2f} 🌌"))

        # =====================
        # ENTRY LOGIC 🚀
        # =====================
        if not self.position:
            # Check cosmic squeeze conditions
            squeeze_on = ((self.keltner_upper[-1] < self.upper_band[-1]) and 
                          (self.keltner_lower[-1] > self.lower_band[-1]))
            
            # Check stellar momentum crossover (replaced backtesting.lib.crossover)
            if squeeze_on and (self.vw_roc[-2] < self.middle_band[-2] and self.vw_roc[-1] > self.middle_band[-1]):
                #