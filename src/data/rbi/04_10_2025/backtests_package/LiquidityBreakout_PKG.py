Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper Moon Dev compliant alternatives:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDITY BREAKOUT STRATEGY �✨

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION 🌌
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data according to Moon Dev specifications
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns with Moon Dev precision ✨
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'bid_volume': 'BidVolume',    # Critical order book metrics
    'ask_volume': 'AskVolume',    # Must exist in source data
    'datetime': 'Date'
}, inplace=True)

# Convert and index datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# =====================
# STRATEGY IMPLEMENTATION 🌗
# =====================
class LiquidityBreakout(Strategy):
    imbalance_period = 20
    volatility_window = 4       # 1-hour volatility (4*15min)
    median_period = 2880        # 30-day median (2880*15min)
    atr_period = 14
    risk_pct = 0.01             # 1% risk per trade
    
    def init(self):
        # 🌗 ORDER BOOK IMBALANCE CALCULATION
        self.raw_imbalance = self.I(lambda df: (df['BidVolume'] - df['AskVolume']) / 
                                  (df['BidVolume'] + df['AskVolume'] + 1e-8), self.data.df)
        
        # 📈 Statistical Bands
        self.imb_ma = self.I(talib.SMA, self.raw_imbalance, self.imbalance_period)
        self.imb_std = self.I(talib.STDDEV, self.raw_imbalance, self.imbalance_period)
        self.upper_band = self.I(lambda x: x[0] + 2*x[1], self.imb_ma, self.imb_std)
        self.lower_band = self.I(lambda x: x[0] - 2*x[1], self.imb_ma, self.imb_std)
        
        # 🌊 Volatility Filter
        self.realized_vol = self.I(talib.ATR, self.data.High, self.data.Low, 
                                 self.data.Close, self.volatility_window)
        self.vol_median = self.I(lambda x: x.rolling(self.median_period).median(),
                               self.realized_vol)
        
        # 🛑 Stop Loss & Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                        self.data.Close, self.atr_period)
        
        # 🪐 Liquidity Zones Detection
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)

    def next(self):
        # 🌑 Daily Loss Circuit Breaker
        if self.equity < 0.95 * 1_000_000:
            print("🌑 MOON DEV ALERT: 5% Daily Loss Limit Triggered! Stopping Trading �")
            self.position.close()
            self.stop()
            
        if self.position:
            # 🏁 Check Exit Conditions
            if self.position.is_long and (self.data.Close[-1] >= self.swing_high[-1] or
                                        (self.raw_imbalance[-2] > self.lower_band[-2] and 
                                         self.raw_imbalance[-1] < self.lower_band[-1])):
                self.position.close()
                print(f"🌕 MOON EXIT LONG at {self.data.Close[-1]:.2f} - Reached Cosmic Resistance! 🚀")
                
            elif self.position.is_short and (self.data.Close[-1] <= self.swing_low