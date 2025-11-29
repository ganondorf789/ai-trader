Here's the complete backtest implementation for the VolSqueezeBreakout strategy:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

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
                crossover(self.data.Close, self.keltner_upper)
            )
            
            # Short entry conditions
            short_cond = (
                self.bbw[-1] < self.bbw_percentile[-1] and
                self.data.Volume[-1] > self.volume_sma50[-1] and
                crossunder(self.data.Close, self.keltner_lower)
            )
            
            if long_cond:
                self.enter_long()
            
            elif short_cond:
                self.enter_short()
        
        # Exit logic
        else:
            if self.position.is_long and crossunder(self.data.Close, self.ema20):
                print(f"🌙 Lunar Profit Taken! 🚀 Closing long at {self.data.Close[-1]:.2f}")
                self.position.close()
            
            elif self.position.is_short and crossover(self.data.Close, self.ema20):
                print(f"🌙 Short Circuit Exit! 🌑 Closing short at {self.data.Close[-1]:.2f}")
                self.position.close()

    def enter_long(self):
        entry_price = self.data.Close[-1]
        stop_loss = self.keltner_lower[-1]
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            print("🌙 Abort Launch! 🚨 Negative risk detected in long entry")
            return
        
        position_size = self