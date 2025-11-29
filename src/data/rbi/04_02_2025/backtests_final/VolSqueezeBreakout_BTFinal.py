Here's the fixed and complete backtest code with all Moon Dev debug improvements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation and cleaning
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VolSqueezeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate Bollinger Bands components
        def calculate_bbw(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return (upper - lower) / middle
        
        self.bbw = self.I(calculate_bbw, self.data.Close)
        self.bbw_percentile = self.I(lambda s: s.rolling(100).quantile(0.1), self.bbw)
        
        # Volume indicators
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, 50)
        
        # Keltner Channel components
        self.ema20 = self.I(talib.EMA, self.data.Close, 20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.keltner_upper = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr20)
        self.keltner_lower = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr20)
        
        print("🌙 MOON DEVS STRATEGY INITIALIZED! 🚀✨ Ready to catch volatility waves!")

    def next(self):
        # Moon-themed debug prints every 500 bars
        if len(self.data) % 500 == 0:
            print(f"🌙 Moon Base Status Update ✨ | Bar: {len(self.data)} | Price: {self.data.Close[-1]:.2f} | BBW: {self.bbw[-1]:.4f} | Vol Ratio: {self.data.Volume[-1]/self.volume_sma50[-1]:.2f}x")
        
        # Trade entry logic
        if not self.position:
            # Long entry conditions
            long_cond = (
                self.bbw[-1] < self.bbw_percentile[-1] and
                self.data.Volume[-1] > self.volume_sma50[-1] and
                (self.data.Close[-2] < self.keltner_upper[-2] and self.data.Close[-1] > self.keltner_upper[-1])
            )
            
            # Short entry conditions
            short_cond = (
                self.bbw[-1] < self.bbw_percentile[-1] and
                self.data.Volume[-1] > self.volume_sma50[-1] and
                (self.data.Close[-2] > self.keltner_lower[-2] and self.data.Close[-1] < self.keltner_lower[-1])
            )
            
            if long_cond:
                # Calculate position size based on risk percentage
                risk_amount = self.equity * self.risk_percent
                stop_distance = abs(self.data.Close[-1] - self.keltner_lower[-1])
                size = risk_amount / stop_distance
                self.buy(size=min(size, 1.0))  # Cap at 100% of equity
                print(f"🌙 LUNAR LIFT-OFF! 🚀 Long entry at {self.data.Close[-1]:.2f} | Size: {size:.2%}")
            
            elif short_cond:
                # Calculate position size based on risk percentage
                risk_amount = self.equity * self.risk_percent
                stop_distance = abs(self.data.Close[-1] - self.keltner_upper[-1