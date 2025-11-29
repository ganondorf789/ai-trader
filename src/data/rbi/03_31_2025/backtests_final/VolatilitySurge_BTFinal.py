Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing implementation:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to proper case
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)

# Convert to datetime and set index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Resample to daily for ATR calculation
daily_data = data.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Calculate daily ATR and shift to avoid lookahead bias
daily_atr = talib.ATR(daily_data['High'], daily_data['Low'], daily_data['Close'], timeperiod=14)
daily_atr = daily_atr.shift(1)

# Merge daily ATR back into 15m data
daily_atr_df = daily_atr.reset_index(name='daily_atr')
daily_atr_df['Date'] = daily_atr_df['Date'].dt.date
data['date'] = data.index.date
data = pd.merge(data, daily_atr_df, left_on='date', right_on='Date', how='left')
data.drop(columns=['date', 'Date'], inplace=True)
data['daily_atr'].ffill(inplace=True)

class VolatilitySurge(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    timeout_bars = 192  # 48 hours in 15m bars (48*4)
    
    def init(self):
        # Bollinger Bands with 20,2
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Bandwidth calculation
        self.bandwidth = (self.bb_upper - self.bb_lower)/self.bb_middle
        self.bandwidth_indicator = self.I(lambda: self.bandwidth, name='Bandwidth')
        
        # 6-month low bandwidth (17280 periods)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth_indicator, timeperiod=17280, name='6M_Low')
        
        # Volume 20-day SMA (1920 periods)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 1920, name='Vol_SMA')
        
        print("🌙✨ Moon Dev Indicators Initialized! ✨🌙")

    def next(self):
        # Moon-themed debug prints
        if len(self.data) % 1000 == 0:
            print(f"🌙 MOON DEV DEBUG | Current Bar: {len(self.data)} | Price: {self.data.Close[-1]:.2f}")
        
        # Check for time-based exit if in position
        if self.position:
            if len(self.position.entry_bars) + self.timeout_bars <= len(self.data):
                print(f"🌑 MOON DEV TIMEOUT EXIT | Bars held: {len(self.data) - self.position.entry_bars[-1]}")
                self.position.close()
                return
        
        if not self.position:
            # Entry conditions
            if (self.bandwidth_indicator[-1] <= self.min_bandwidth[-1] and 
                self.data.Volume[-1] > self.volume_sma[-1] and
                not np.isnan(self.data.daily_atr[-1])):
                
                atr_entry = self.data.daily_atr[-1]
                if atr_entry <= 0:
                    print("🚨 MOON DEV ALERT: Invalid ATR Value!")
                    return
                
                # Risk-based position sizing (units)
                risk_amount = self.equity * self.risk_pct
                position_size