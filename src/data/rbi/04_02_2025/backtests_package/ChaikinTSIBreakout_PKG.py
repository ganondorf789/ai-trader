Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper Moon Dev themed implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR CHAIKIN TSI BREAKOUT STRATEGY 🚀

# REQUIRED IMPORTS
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# DATA PREPARATION
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class ChaikinTSIBreakout(Strategy):
    def init(self):
        # 🌙 INDICATOR CALCULATIONS USING TA-LIB & PANDAS_TA
        # Chaikin Money Flow
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20, name='CMF')
        
        # True Strength Index with Signal Line ✨
        def calculate_tsi(close):
            tsi = ta.tsi(close, length=25, signal=13)
            return tsi.iloc[:,0], tsi.iloc[:,1]
        self.tsi_line, self.tsi_signal = self.I(calculate_tsi, self.data.Close, name=['TSI', 'TSI_SIGNAL'])
        
        # Donchian Channels 🌙
        self.donchian_upper = self.I(talib.MAX, self.data.High, 20, name='DON_UPPER')
        self.donchian_lower = self.I(talib.MIN, self.data.Low, 20, name='DON_LOWER')
        self.donchian_width = self.I(lambda h,l: talib.MAX(h,20) - talib.MIN(l,20), 
                                 self.data.High, self.data.Low, name='DON_WIDTH')
        
        # Parabolic SAR 🛑
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                        acceleration=0.02, maximum=0.2, name='SAR')
        
        print("🌙✨ MOON DEV STRATEGY INITIALIZED WITH LUNAR POWER! 🚀")

    def next(self):
        # 🌙 CORE LOGIC EXECUTION
        if len(self.cmf) < 2 or len(self.tsi_line) < 2 or len(self.donchian_width) < 5:
            return

        # LONG ENTRY CONDITIONS ✨
        cmf_bullish = self.cmf[-1] > 0 and self.cmf[-2] <= 0
        tsi_cross = (self.tsi_line[-2] < self.tsi_signal[-2] and self.tsi_line[-1] > self.tsi_signal[-1])  # Bullish crossover replacement
        donchian_cond = all(self.donchian_width[i] > self.donchian_width[i+1] for i in range(-5,-1))
        price_break = self.data.Close[-1] > self.donchian_upper[-1]
        
        if cmf_bullish and tsi_cross and donchian_cond and price_break and not self.position:
            # 🌙 RISK CALCULATION
            risk_pct = 0.01  # 1% risk per trade
            risk_amount = self.equity * risk_pct
            entry = self.data.Close[-1]
            stop_loss = entry * 0.98
            risk_per_unit = entry - stop_loss
            
            if risk_per_unit > 0:
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    self.buy(size=size)
                    print(f"🌙🚀 BULLISH BREAKOUT! Long {size} units @ {entry:.2f} ✨")

        # SHORT ENTRY CONDITIONS 🌙
        cmf_bearish = self.cmf[-1] < 0 and self.cmf[-2] >=