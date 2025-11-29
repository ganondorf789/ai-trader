Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names - Moon Dev style consistency ✨
data.columns = data.columns.str.strip().str.lower()

# Drop unnamed columns - keeping things tidy 🌙
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')

# Standardize column names for backtesting library
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})

# Ensure proper datetime format - Moon Dev precision 🌙
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

class VolatilitySpikeReversal(Strategy):
    def init(self):
        # Calculate indicators using TA-Lib - Moon Dev indicator setup ✨
        close = self.data.Close
        
        # Bollinger Bands with proper indicator wrapping
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda x: x, upper, name='Upper Band')
        self.lower_band = self.I(lambda x: x, lower, name='Lower Band')
        
        # ADX with proper parameter passing
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, close, timeperiod=14, name='ADX')
        
        # Volume SMA with proper indicator wrapping
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=30, name='Volume SMA')

    def next(self):
        # Wait for sufficient data - Moon Dev patience 🌙
        if len(self.data.Close) < 30:
            return
        
        # Moon Dev Debug Prints 🌙✨
        print(f"\n🌙 Moon Dev Analysis - Bar {len(self.data)}:")
        print(f"   Current Close: {self.data.Close[-1]:.2f}")
        print(f"   Upper Band: {self.upper_band[-1]:.2f}")
        print(f"   Lower Band: {self.lower_band[-1]:.2f}")
        print(f"   ADX: {self.adx[-1]:.2f}")
        print(f"   Volume Ratio: {self.data.Volume[-1]/self.volume_sma[-1]:.2f}x")

        # Long entry conditions - unchanged strategy logic
        if (not self.position and
            self.data.Low[-1] <= self.lower_band[-1] and
            self.adx[-1] < 20 and
            self.data.Volume[-1] > self.volume_sma[-1]):
            
            # Risk management calculations - Moon Dev precision ✨
            risk_pct = 0.01  # 1% risk
            entry_price = self.data.Close[-1]
            stop_loss = self.lower_band[-1]
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share > 0:
                position_size = (self.equity * risk_pct) / risk_per_share
                position_size = int(round(position_size))  # Ensuring whole units
                
                # Moon Dev Position Sizing ✨
                print(f"\n🚀 Moon Dev Trade Signal!")
                print(f"   Entry Price: {entry_price:.2f}")
                print(f"   Stop Loss: {stop_loss:.2f}")
                print(f"   Calculated Size: {position_size} units")
                
                self.buy(size=position_size)

        # Exit conditions - unchanged strategy logic
        if self.position.is_long:
            # Take profit at upper band
            if self.data.High[-1] >= self.upper_band[-1]:
                print(f"\n💰 Moon Dev Profit Capture!")
                print(f"   Upper Band Hit: {self.upper_band[-1]:.2f}")
                self.position.close()
            
            # Stop loss at lower band
            elif self.data.Close[-1] < self.lower_band[-1]:
                print(f"\n🌧️ Moon Dev Risk Protection!")
                print(f"   Lower Band Breach: {self.lower_band[-1]:.2f}")
                self.position.close()

# Execute backtest - Moon Dev