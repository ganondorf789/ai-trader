I'll fix the code to remove all backtesting.lib imports and replace any crossover functions with proper implementations. Here's the corrected version with Moon Dev themed improvements:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
# 🚀 PACKAGE AI-ENHANCED VERSION 🚀

# ===== REQUIRED IMPORTS =====
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# ===== DATA PREPARATION =====
# Load and clean data with Moon Dev standards 📊✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping for Moon Dev backtesting framework 🌙📈
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ===== VOLSQUEEZEREVERSAL STRATEGY CLASS =====
class VolSqueezeReversal(Strategy):
    risk_percent = 0.02  # 2% risk per trade 🌙💰
    
    def init(self):
        # 🌙 INDICATOR CALCULATION USING TA-LIB 🌙
        # Bollinger Bands components 📊
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        
        # 3-month low of BB Width (8640 periods for 15m data) 📅
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=8640)
        
        # Volume surge detection 🔊
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 1920)  # 20-day SMA
        self.volume_surge = self.I(lambda v, ma: v > 1.5 * ma, self.data.Volume, self.volume_ma)
        
        self.peak = None  # Track highest high after entry 🏔️

    def next(self):
        # Moon Dev progress tracking 🌙⏳
        if len(self.data) % 1000 == 0:
            print(f"🌙 MOON DEV PROGRESS: Processing bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} ✨")
        
        # ENTRY LOGIC 🚪
        if not self.position:
            # Check volatility squeeze condition 🔍
            bb_low_condition = self.bb_width[-1] == self.bb_width_low[-1]
            # Check volume surge condition 🔊
            volume_condition = self.volume_surge[-1] > 0
            
            if bb_low_condition and volume_condition:
                # Moon Dev risk calculation 🌙🧮
                entry_price = self.data.Close[-1]
                position_size = int(round((self.equity * self.risk_percent) / (entry_price * self.risk_percent)))
                
                # Moon Dev themed entry alert 🚀
                print(f"\n🚀 MOON DEV ENTRY SIGNAL 🚀")
                print(f"📉 BB Width: {self.bb_width[-1]:.4f} (3-month low!)")
                print(f"📈 Volume Surge: {self.data.Volume[-1]:.2f} vs MA {self.volume_ma[-1]:.2f}")
                print(f"💰 Buying {position_size} units at {entry_price:.2f}\n")
                
                self.buy(size=position_size)
                self.peak = self.data.High[-1]  # Initialize peak tracker
                
        # EXIT LOGIC 🏃
        else:
            # Update trailing peak 🌊
            self.peak = max(self.peak, self.data.High[-1])
            
            # Check 2%