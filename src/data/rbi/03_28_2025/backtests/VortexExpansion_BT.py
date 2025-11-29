To address this problem, we need to create a backtesting implementation for the VortexExpansion strategy using the backtesting.py framework. The strategy involves using the Vortex Indicator and ATR for entry and exit signals, along with proper risk management and position sizing.

### Approach
1. **Data Handling**: Load and preprocess the data to ensure proper column names and formatting.
2. **Indicator Calculation**: Compute the Vortex Indicator (VI+ and VI-) using pandas_ta and the ATR values using TA-Lib, ensuring they are added as indicators using the `self.I()` wrapper.
3. **Entry/Exit Logic**: Implement the logic for entering long positions when VI+ crosses above VI- and ATR is expanding, and exiting when VI+ crosses below VI- or ATR contracts.
4. **Risk Management**: Calculate position sizes based on a fixed percentage of equity risk and round to the nearest integer to comply with backtesting.py's requirements.
5. **Execution**: Run the backtest with an initial cash amount of 1,000,000 and print the results.

### Solution Code
```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexExpansion(Strategy):
    def init(self):
        # Calculate Vortex Indicator using pandas_ta
        df = self.data.df.rename(columns={
            'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close'
        })
        vortex = df.ta.vortex(length=14)
        self.vi_plus = self.I(lambda: vortex['VORTICSm_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTICSs_14'], name='VI-')
        
        # Calculate ATR(2) and its 10-period SMA
        self.atr2 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 2, name='ATR2')
        self.atr10_sma = self.I(talib.SMA, self.atr2, 10, name='ATR10_SMA')
    
    def next(self):
        # Wait until all indicators are valid
        if len(self.data) < 25:
            return
        
        # Current values
        vi_plus = self.data['VI+'][-1]
        vi_minus = self.data['VI-'][-1]
        atr2 = self.data['ATR2'][-1]
        atr10_sma = self.data['ATR10_SMA'][-1]
        
        # Check crossovers
        vi_cross_above = vi_plus > vi_minus and self.data['VI+'][-2] <= self.data['VI-'][-2]
        vi_cross_below = vi_plus < vi_minus and self.data['VI+'][-2] >= self.data['VI-'][-2]
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
            
            print(f"🌙✨ MOON DEV LONG SIGNAL ✨ Entry: {entry_price:.