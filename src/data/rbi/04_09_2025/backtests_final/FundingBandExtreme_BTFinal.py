Here's the fully debugged code with Moon Dev themed improvements while preserving the original strategy logic:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data with Moon Dev cosmic checks 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    print("🌙 Data successfully loaded from cosmic archives!")
except Exception as e:
    raise Exception(f"🚨 MOON DEV ALERT: Data loading failed - {str(e)}")

# Clean column names with lunar precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Critical funding rate verification
if 'funding' not in data.columns:
    raise ValueError("🌙 CRITICAL: Missing funding rate column - lunar strategy cannot orbit without it!")
data['Funding'] = data['funding'].astype(float)
print("✨ Funding rates calibrated for zero-gravity trading!")

class FundingBandExtreme(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    stop_loss_pct = 0.02  # 2% stop loss
    long_threshold = 0.0001  # Positive funding threshold
    short_threshold = -0.0001  # Negative funding threshold
    funding_neutral = 0.00005  # Neutral zone threshold
    contraction_pct = 0.2  # 20% band contraction exit

    def init(self):
        # Bollinger Bands with lunar precision 🌙
        def bbands_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
            
        def bbands_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
            
        self.upper_band = self.I(bbands_upper, self.data.Close)
        self.lower_band = self.I(bbands_lower, self.data.Close)
        self.funding_rate = self.data.Funding
        print("🌙 Technical indicators launched into orbit!")

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Cosmic entry conditions ✨
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            current_funding = self.funding_rate[-1]
            
            # Short entry with lunar gravity assist
            if current_high >= self.upper_band[-1] and current_funding > self.long_threshold:
                sl_price = price * (1 + self.stop_loss_pct)
                risk_per_share = sl_price - price
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:  # Ensure we don't trade zero units
                        self.sell(size=size, sl=sl_price, tag='short_entry')
                        print(f"🌙 SHORT FIRED! Price: {price:.2f} | Size: {size} units | Funding: {current_funding:.5f} ✨")
            
            # Long entry with rocket boosters 🚀
            elif current_low <= self.lower_band[-1] and current_funding < self.short_threshold:
                sl_price = price * (1 - self.stop_loss_pct)
                risk_per_share = price - sl_price
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:  # Position size validation
                        self.buy(size=size, sl=sl_price, tag='long_entry')
                        print(f"🚀 LONG LAUNCHED! Price: {price:.2f} | Size: {size} units | Funding: {current_funding:.5f} 🌙")
        
        else:
            # Exit conditions with cosmic precision
            position = self.position
            entry_width = (self.upper_band[-1] - self.lower_band[-1