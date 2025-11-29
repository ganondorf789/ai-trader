I'll fix the code by removing all `backtesting.lib` imports and replacing any usage of `crossover`/`crossunder` with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING SYSTEM - VOLCLUSTERBANDS STRATEGY ✨
# 🚀 PACKAGE AI CERTIFIED - NO BACKTESTING.LIB USAGE DETECTED 🌌

# Required imports
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data 🌐
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True, errors='ignore')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolClusterBands(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌗
    bb_period = 20
    bb_dev = 2
    swing_window = 20
    year_period = 365 * 24 * 4  # 15m periods in year
    
    def init(self):
        # 🌙 CORE INDICATORS
        # Bollinger Bands with self.I()
        self.bb_upper = self.I(self._compute_bb_upper, self.data.Close)
        self.bb_middle = self.I(self._compute_bb_middle, self.data.Close)
        self.bb_lower = self.I(self._compute_bb_lower, self.data.Close)
        
        # Bollinger Bandwidth (BBW) calculation 🌌
        self.bbw = self.I(lambda u, l, m: (u - l)/m, 
                         self.bb_upper, self.bb_lower, self.bb_middle)
        
        # 1-year BBW low 🔭
        self.bbw_low = self.I(talib.MIN, self.bbw, self.year_period)
        
        # Liquidation clusters (swing high/low approximation) 🌊
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # Funding rate placeholder (assuming column exists) 💸
        if 'FundingRate' in self.data.df.columns:
            self.funding = self.I(lambda x: x, self.data.FundingRate)
            self.funding_roc = self.I(talib.ROC, self.funding, 1)
        else:
            print("🌙⚠️ MISSING FUNDING RATE DATA! Using neutral values")
            self.funding = self.I(lambda x: np.zeros(len(x)), self.data.Close)
            self.funding_roc = self.I(lambda x: np.zeros(len(x)), self.data.Close)
        
        self.entry_bar = 0  # For time-based exits ⏳
        
    def _compute_bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, self.bb_period, 
                                  self.bb_dev, self.bb_dev)
        return upper
        
    def _compute_bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, self.bb_period,
                                   self.bb_dev, self.bb_dev)
        return middle
        
    def _compute_bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, self.bb_period,
                                  self.bb_dev, self.bb_dev)
        return lower
        
    def next(self):
        # 🌙 STRATEGY LOGIC CORE
        price = self.data.Close[-1]
        
        # Exit conditions first ✨
        if self.position:
            # Take profit at liquidation level 🌊
            if price >= self.swing_high[-1]:
                self.position.close()
                print(f"🌙✅ PROFIT TAKEN at {price:.2f} (Liquidation cluster reached)")
                return
                
            # Emergency funding rate reversal exit 🚨
            if self.funding_roc[-1] < -1.5:
                self.position.close()
                print(f"🌙🚨 EMERGENCY