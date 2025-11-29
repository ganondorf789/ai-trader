Here's the complete backtesting implementation for the LiquidityFade strategy:

```python
# 🌙 Moon Dev's Liquidity Fade Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

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
        self.funding_series = self.data.df['funding_rate']
        
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
                print(f"🌙 SHORT SIGNAL! CMO: {cmo:.1f} | Funding: {funding:.4f} | Size: {position_size}")
                
        elif cmo < 10 and funding < -0.0001 and price <= swing_l * (1 + self.liquidity_proximity):
            # Long Entry Logic
            sl_price = swing_l - atr * 1.5
            risk_per_share = price - sl_price
            position_size = self.calculate_size(risk_per_share)
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price,
                        tp=price + atr * 2)
                print(f"🚀 LONG SIGNAL! CMO: {cmo:.1f} | Funding: {funding:.4f} | Size: {position_size}")
    
    def calculate_size(self, risk_per_share):
        """Moon-sized Risk Management 🌜"""
        if risk_per_share <= 0:
            return 0
        risk_amount = self.equity * self.risk_pct
        size = risk_amount / risk_per_share
        return int(round(size))

# ========================
# BACKTEST EXECUTION
# ========================
bt = Backtest(data, LiquidityFade, cash=1_000_000, commission=.002)
stats = bt.run()
print("\n🌕🌖🌗🌘🌑🌒🌓🌔🌕")
print("MOON DEV BACKTEST RESULTS:")
print(stats)
print("\