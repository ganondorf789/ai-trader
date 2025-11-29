Here's the implementation of the SkewInversion strategy for backtesting.py:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR SKEWINVERSION STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================
# DATA PREPARATION 🌍
# =====================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'

# Load and clean data 🌙
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌌
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY IMPLEMENTATION 🚀
# =====================
class SkewInversion(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌙
    
    def init(self):
        # 🌙 VIX Term Structure Indicators
        self.vix_front = self.I(lambda: self.data.df['vix_front'], name='VIX Front')
        self.vix_back = self.I(lambda: self.data.df['vix_back'], name='VIX Back')
        self.term_structure = self.I(talib.SUB, self.vix_front, self.vix_back, name='Term Structure')
        self.term_sma = self.I(talib.SMA, self.term_structure, 10, name='Term SMA')
        
        # 🌌 SPX Skew Indicators
        self.spx_skew = self.I(lambda: self.data.df['spx_skew'], name='SPX Skew')
        self.skew_mean = self.I(talib.SMA, self.spx_skew, 30, name='Skew Mean')
        self.skew_std = self.I(talib.STDDEV, self.spx_skew, 30, name='Skew Std')
        self.upper_band = self.I(talib.ADD, self.skew_mean, 
                                self.I(talib.MULT, self.skew_std, 2), name='+2σ')
        self.three_sigma = self.I(talib.ADD, self.skew_mean,
                                 self.I(talib.MULT, self.skew_std, 3), name='+3σ')

    def next(self):
        price = self.data.Close[-1]
        
        # 🌙✨ Moon Dev Debug Prints
        print(f"\n🌙 DATE: {self.data.index[-1]} | PRICE: {price:.2f}")
        print(f"📈 VIX Structure: {self.term_structure[-1]:.2f} (SMA: {self.term_sma[-1]:.2f})")
        print(f"🌌 SPX Skew: {self.spx_skew[-1]:.2f} vs Bands: {self.upper_band[-1]:.2f}/{self.three_sigma[-1]:.2f}")

        # Entry Logic 🌙
        if not self.position:
            # Check inversion and skew conditions
            term_inverted = self.term_structure[-1] > 0
            skew_extreme = self.spx_skew[-1] > self.upper_band[-1]
            
            if term_inverted and skew_extreme:
                # Moon-themed position sizing 🌙
                risk_amount = self.risk_per_trade * self.equity
                atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 14)[-1]
                position_size = int(round(risk_amount / (2 * atr))) if atr > 0 else 0
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🚀🌕 MOON DEV ENTRY! Buying {position_size} shares")
                    print(f"🎯 REASON: VIX inversion + Skew > 2σ ({self.spx_skew[-1]:.2f} > {self.upper_band[-1]:.2f})")

        # Exit Logic 🌙
        else:
            current_size = self.position.size
            
            # Stop Loss Conditions 🌙
            stop