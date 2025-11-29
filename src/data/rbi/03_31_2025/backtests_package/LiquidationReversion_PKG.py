Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT - LIQUIDATION REVERSION STRATEGY 🚀

# 1. IMPORT REQUIRED LIBRARIES
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# 2. DATA PREPARATION
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data with Moon Dev standards 🌙
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py requirements with proper case
required_columns = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=required_columns, inplace=True)

# 3. STRATEGY IMPLEMENTATION
class LiquidationReversion(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 INDICATOR CALCULATION USING TA-LIB
        # Funding Rate Analysis
        self.funding = self.data.df['funding_rate']
        self.funding_z = self.I(ta.zscore, self.funding, length=100, name='FUNDING_Z')
        
        # Volatility Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, 50, name='ATR_MA')
        
        # Reversal Indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 20, 2, 2, 0,
            name=['BB_UP', 'BB_MID', 'BB_LOW']
        )
        
        print("🌙 STRATEGY INITIALIZED | Moon Dev Ready for Launch! 🚀")

    def next(self):
        # 🌙 MOON DEV DEBUG PRINTING
        if len(self.data) % 100 == 0:
            print(f"🌙 PROGRESS CHECK | Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | Funding Z: {self.funding_z[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        # Exit conditions - volatility expansion
        if self.position and self.atr[-1] > 1.5*self.atr_ma[-1]:
            self.position.close()
            print(f"🌙💎 VOLATILITY EXIT | Price: {self.data.Close[-1]:.2f} | Profit: {self.position.pl:.2f}")
        
        # Entry logic when flat
        if not self.position:
            # Long conditions
            long_trigger = (
                (self.funding_z[-1] < -2) and
                (self.atr[-1] > 2*self.atr_ma[-1]) and
                (self.atr[-1] < 1.5*self.atr_ma[-1]) and
                (self.rsi[-1] < 30)
            )
            
            # Short conditions
            short_trigger = (
                (self.funding_z[-1] > 2) and
                (self.atr[-1] > 2*self.atr_ma[-1]) and
                (self.atr[-1] < 1.5*self.atr_ma[-1]) and
                (self.rsi[-1] > 70)
            )
            
            # 🌙 Execute long entry
            if long_trigger:
                sl = self.data.Close[-1] - 1.5*self.atr[-1]
                tp = self.data.Close[-1] + 2*self.atr[-1]
                risk_amount = self.risk_per_trade * self.equity
                size = int(round(risk_amount / (self.data.Close[-1] - sl)))
                
                self.buy(size=size, sl