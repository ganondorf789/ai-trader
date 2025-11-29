Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and handle data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
assert 'funding' in data.columns, "🌙 CRITICAL: Missing funding rate column - check data source!"
data['Funding'] = data['funding']

class FundingBandExtreme(Strategy):
    risk_pct = 0.01
    stop_loss_pct = 0.02
    long_threshold = 0.0001
    short_threshold = -0.0001
    funding_neutral = 0.00005
    contraction_pct = 0.2

    def init(self):
        # Bollinger Bands indicators using talib
        def bbands_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        def bbands_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
            
        self.upper_band = self.I(bbands_upper, self.data.Close)
        self.lower_band = self.I(bbands_lower, self.data.Close)
        self.funding_rate = self.data.Funding

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Entry logic
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            current_funding = self.funding_rate[-1]
            
            # Short entry
            if current_high >= self.upper_band[-1] and current_funding > self.long_threshold:
                sl_price = price * (1 + self.stop_loss_pct)
                risk_per_share = sl_price - price
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    self.sell(size=size, tag='short_entry')
                    print(f"🌙 SHORT FIRED! Price: {price:.2f} | Funding: {current_funding:.5f} ✨")
            
            # Long entry
            elif current_low <= self.lower_band[-1] and current_funding < self.short_threshold:
                sl_price = price * (1 - self.stop_loss_pct)
                risk_per_share = price - sl_price
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    self.buy(size=size, tag='long_entry')
                    print(f"🚀 LONG LAUNCHED! Price: {price:.2f} | Funding: {current_funding:.5f} 🌙")
        
        else:
            # Exit logic
            position = self.position
            entry_width = self.upper_band[-1] - self.lower_band[-1] if position.is_short else self.upper_band[-2] - self.lower_band[-2]
            current_width = self.upper_band[-1] - self.lower_band[-1]
            
            # Volatility contraction
            if current_width <= entry_width * (1 - self.contraction_pct):
                position.close()
                print(f"🌙 VOLATILITY COLLAPSE! Closed {position.size} units 🌊")
            
            # Funding normalization
            elif abs(self.funding_rate[-1]) <= self.funding_neutral:
                position.close()
                print(f"✨ FUNDING NORMALIZED! Rate: {self.funding_rate[-1]:.5f} 🌙")
            
            # Stop loss check
            elif position.is_short and self.data.High[-1] >= price * (1 + self.stop_loss_pct):
                position.close()
                print(f"🛑 SHORT STOPPED OUT! Loss