Here's the debugged version of your backtest code with Moon Dev themed improvements 🌙✨:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names - Moon Dev Standardization Protocol 🌙
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map required columns with Moon Dev Precision ✨
required_columns = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'funding_rate',
    'open_interest': 'open_interest'
}

for orig, new in required_columns.items():
    if orig in data.columns:
        data.rename(columns={orig: new}, inplace=True)

# Moon Dev Data Validation Check 🌕
missing_cols = [col for col in required_columns.values() if col not in data.columns]
if missing_cols:
    print(f"🌑 WARNING: Missing critical columns - {missing_cols}")

class LiquidationPulse(Strategy):
    risk_pct = 0.01  # 1% risk per trade (Moon Dev Approved Risk Level 🌙)
    oi_lookback = 2  # Consecutive OI increases required
    cluster_period = 20  # Periods for cluster identification
    
    def init(self):
        # Precompute indicators using TA-Lib with Moon Dev Precision ✨
        self.funding_rate = self.I(lambda x: x, self.data.df['funding_rate'], name='Funding Rate')
        self.open_interest = self.I(lambda x: x, self.data.df['open_interest'], name='Open Interest')
        
        # Calculate OI changes using TSF as proxy for difference
        self.oi_diff = self.I(talib.TSF, self.open_interest, timeperiod=1, name='OI Difference')
        self.consecutive_oi_increases = self.I(talib.SUM, 
            (self.oi_diff > np.roll(self.oi_diff, 1)).astype(int), 
            timeperiod=self.oi_lookback,
            name='Consecutive OI Increases')
        
        # Cluster identification - Moon Dev Cluster Detection System 🌙
        self.low_cluster = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_period, name='Low Cluster')
        self.high_cluster = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_period, name='High Cluster')
        
        # Track initial risk for trailing stops
        self.initial_risk = None

    def next(self):
        price = self.data.Close[-1]
        
        if self.position:
            # Moon Dev Trailing Stop Logic 🌙✨
            if price >= self.position.entry_price + 1.5 * self.initial_risk:
                self.position.close()
                print(f"🌕 MOON DEV TRAILING STOP ACTIVATED: Reached 1.5R at {price:.2f}!")
                return
                
            # Check exit conditions with Moon Dev Precision ✨
            if self.funding_rate[-1] >= 0:
                self.position.close()
                print(f"🌑 MOON DEV FUNDING RATE EXIT: Turned positive ({self.funding_rate[-1]:.6f})")
            elif self.oi_diff[-1] < self.oi_diff[-2]:
                self.position.close()
                print(f"🌒 MOON DEV OI DECLINE EXIT: Open interest decreased")
            elif price >= self.high_cluster[-1]:
                self.position.close()
                print(f"🌓 MOON DEV RESISTANCE EXIT: Reached high cluster {self.high_cluster[-1]:.2f}")
        else:
            # Entry conditions check - Moon Dev Entry Scanner Active 🌙
            entry_conditions = (
                self.funding_rate[-1] < 0,
                self.consecutive_oi_increases[-1] >= self.oi_lookback,
                price > self.low_cluster[-1]
            )
            
            if all(entry_conditions):
                # Moon Dev Risk Calculation System 🌙✨
                sl_price = self.low_cluster[-1] * 0.995
                risk_per_share = price - sl_price
                risk_amount =