Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed implementation:

```python
# 🌙 Moon Dev's VoltaicBreakout Backtest Implementation ✨
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# ========================
# DATA PREPARATION 🌙 CLEANSE
# ========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ========================
# VOLTAIC BREAKOUT STRATEGY 🌙
# ========================
class VoltaicBreakout(Strategy):
    def init(self):
        print("🌙 Initializing Moon Dev's Voltaic Breakout...")
        
        # ===== BOLLINGER BANDS & BANDWIDTH =====
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Bandwidth calculation
        self.bb_width = self.I(lambda u, l, m: (u - l)/m*100, 
            self.upper, self.lower, self.middle)
        
        # Bandwidth percentile (100-period lookback)
        self.bbw_pct = self.I(lambda x: ta.percentile(x, 100), self.bb_width)
        
        # ===== VOLUME SURGE DETECTION =====
        self.vol_ma50 = self.I(talib.SMA, self.data.Volume, 4800)  # 50-day MA (15m)
        
        # ===== BREAKOUT LEVELS =====
        self.d3_high = self.I(talib.MAX, self.data.High, 288)  # 3-day high (15m)
        self.d3_low = self.I(talib.MIN, self.data.Low, 288)    # 3-day low
        
        # ===== DYNAMIC STOP LEVELS =====
        self.swing_low = self.I(talib.MIN, self.data.Low, 96)  # Daily swing low
        self.swing_high = self.I(talib.MAX, self.data.High, 96) # Daily swing high
        
        print("✨ Strategy Components Activated: BBANDS | Volume Surge | Swing Levels")

    def next(self):
        # Skip first 5000 bars for indicator warmup
        if len(self.data) < 5000:
            return
        
        # Current values
        price = self.data.Close[-1]
        bbw_pct = self.bbw_pct[-1]
        vol_ratio = self.data.Volume[-1]/self.vol_ma50[-1]
        
        # ===== ENTRY LOGIC 🌙 =====
        if not self.position:
            # Long Entry Conditions
            if (bbw_pct < 20 and 
                vol_ratio > 1 and 
                price > self.d3_high[-1]):
                
                # Risk management 🌙
                sl = self.swing_low[-1]
                risk_pct = 0.01  # 1% risk
                risk_amount = self.equity * risk_pct
                risk_per_share = price - sl
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount/risk_per_share))
                    tp = price + 2*(price - sl)  # 2:1 RR
                    
                    self.buy(size=position_size, sl=sl, tp=tp)
                    print(f"🚀🌙 LONG SIGNAL | Size: {position_size} | Entry: {price:.2f}")
            
            # Short Entry Conditions    
            elif (bbw_pct < 20 and 
                  vol_ratio > 1 and 
                  price < self.d3_low[-1]):
                  
                sl = self.swing_high[-1]
                risk_pct = 0.01
                risk_amount = self.equity * risk_pct
                risk_per_share = sl - price
                
                if risk_per_share >