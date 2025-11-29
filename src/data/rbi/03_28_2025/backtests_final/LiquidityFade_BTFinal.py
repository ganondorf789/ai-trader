I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Liquidity Fade Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========================
# DATA PREPARATION
# ========================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'

# Load and clean data
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping for backtesting.py
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_map, inplace=True)

# ========================
# STRATEGY IMPLEMENTATION
# ========================
class LiquidityFade(Strategy):
    timeperiod_cmo = 20
    swing_period = 20
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    liquidity_proximity = 0.005  # 0.5% threshold
    
    def init(self):
        # 🌙 Core Indicators
        self.cmo = self.I(talib.CMO, self.data.Close, self.timeperiod_cmo)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, self.atr_period)
        
        # Funding Rate (assuming column exists in data)
        if 'funding_rate' in self.data.df.columns:
            self.funding_series = self.data.df['funding_rate']
        else:
            self.funding_series = pd.Series(0, index=self.data.df.index)
        
        # Moon Dev Debug Initialization
        print("🌙 Lunar Indicators Initialized:")
        print(f"- CMO Period: {self.timeperiod_cmo}")
        print(f"- Swing Period: {self.swing_period}")
        print(f"- ATR Period: {self.atr_period}")
        print("✨ Ready for liftoff! 🚀")
        
    def next(self):
        # Avoid multiple positions
        if self.position:
            return

        # 🌟 Current Values
        price = self.data.Close[-1]
        cmo = self.cmo[-1]
        funding = self.funding_series[-1]
        swing_h = self.swing_high[-1]
        swing_l = self.swing_low[-1]
        atr = self.atr[-1]
        
        # 🚨 Entry Conditions
        if cmo > 90 and funding < -0.0001 and price >= swing_h * (1 - self.liquidity_proximity):
            # Short Entry Logic
            sl_price = swing_h + atr * 1.5
            risk_per_share = sl_price - price
            position_size = self.calculate_size(risk_per_share)
            
            if position_size > 0:
                self.sell(size=position_size, sl=sl_price, 
                         tp=price - atr * 2)
                print(f"🌑 NEW MOON SHORT! CMO: {cmo:.1f} | Funding: {funding:.4f} | Size: {position_size:.2f}")
                
        elif cmo < 10 and funding < -0.0001 and price <= swing_l * (1 + self.liquidity_proximity):
            # Long Entry Logic
            sl_price = swing_l - atr * 1.5
            risk_per_share = price - sl_price
            position_size = self.calculate_size(risk_per_share)
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price,
                        tp=price + atr * 2)
                print(f"🌕 FULL MOON LONG! CMO: {cmo:.1f} | Funding: {funding:.4f} | Size: {position_size:.2f}")
    
    def calculate_size(self, risk_per_share):
        """Moon-sized Risk Management 🌜"""
        if risk_per_share <= 0: