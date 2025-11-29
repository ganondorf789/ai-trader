I'll fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Moon Dev AI Package Initialization 🌙
print("🌙✨ Initializing Moon Dev's Vortex Expansion Strategy Package ✨🌙")

# Load and preprocess data
print("🌙 Loading celestial market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
print("🌙✨ Sanitizing data columns with lunar precision...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set as index
print("🌙 Aligning temporal dimensions...")
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexExpansion(Strategy):
    def init(self):
        print("🌙✨ Initializing Vortex Expansion Strategy with Moon Dev precision...")
        
        # Calculate Vortex Indicator using pandas_ta
        df = self.data.df.rename(columns={
            'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close'
        })
        vortex = df.ta.vortex(length=14)
        self.vi_plus = self.I(lambda: vortex['VORTICSm_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTICSs_14'], name='VI-')
        
        # Calculate ATR(2) and its 10-period SMA using talib
        print("🌙 Calculating celestial indicators...")
        self.atr2 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=2, name='ATR2')
        self.atr10_sma = self.I(talib.SMA, self.atr2, timeperiod=10, name='ATR10_SMA')
    
    def next(self):
        # Wait until all indicators are valid
        if len(self.data) < 25:
            return
        
        # Current values
        vi_plus = self.vi_plus[-1]
        vi_minus = self.vi_minus[-1]
        atr2 = self.atr2[-1]
        atr10_sma = self.atr10_sma[-1]
        
        # Check crossovers using proper array indexing
        vi_cross_above = (self.vi_plus[-2] <= self.vi_minus[-2]) and (vi_plus > vi_minus)
        vi_cross_below = (self.vi_plus[-2] >= self.vi_minus[-2]) and (vi_plus < vi_minus)
        atr_expanding = atr2 > atr10_sma
        atr_contracting = atr2 < atr10_sma
        
        # Risk management parameters
        risk_pct = 0.01  # 1% of equity
        entry_price = self.data.Close[-1]
        
        # Long entry logic
        if not self.position and vi_cross_above and atr_expanding:
            stop_loss_pct = 0.01  # 1% stop loss
            stop_loss = entry_price * (1 - stop_loss_pct)
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return
            
            risk_amount = self.equity * risk_pct
            position_size = int(round(risk_amount / risk_per_share))
            position_size = max(1, position_size)  # Minimum 1 unit
            
            print(f"🌙✨ MOON DEV LONG SIGNAL ✨ Entry: {entry_price:.2f}, Size: {position_size} units, SL: {stop_loss:.2f}")
            self.buy(size=position_size, sl=stop_loss)
        
        # Exit logic
        elif self.position.is_long and (vi_cross_below or atr_contracting):
            print(f"🌙✨ MOON DEV EXIT SIGNAL ✨ Closing position at {entry_price:.2f}")
            self.position.close()

# Run backtest
print("