```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation Ritual 🌙✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse columns with lunar precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align cosmic coordinates (proper column mapping)
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set temporal index for time continuum synchronization
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSurgeBreakout(Strategy):
    def init(self):
        # Cosmic Energy Indicators 🌌
        # Bollinger Bands (2σ) - Entry Signals
        self.upper_2std = self.I(lambda: talib.BBANDS(self.data.Close, 20, 2, 2, 0)[0], name='UPPER_2STD')
        self.lower_2std = self.I(lambda: talib.BBANDS(self.data.Close, 20, 2, 2, 0)[2], name='LOWER_2STD')
        
        # Bollinger Bands (1σ) - Exit Signals
        self.upper_1std = self.I(lambda: talib.BBANDS(self.data.Close, 20, 1, 1, 0)[0], name='UPPER_1STD')
        self.lower_1std = self.I(lambda: talib.BBANDS(self.data.Close, 20, 1, 1, 0)[2], name='LOWER_1STD')
        
        # Volume Surge Detector 📈
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOLUME_SMA')

    def next(self):
        # Wait until cosmic alignment (20-period warmup)
        if len(self.data) < 20:
            return

        # Current celestial readings 🌠
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        volume_avg = self.volume_sma[-1]

        # Moon Dev Trading Protocol 🚀
        if not self.position:
            # Long Entry: Price breaches upper 2σ with volume surge
            if price > self.upper_2std[-1] and volume >= 1.5 * volume_avg:
                moon_units = int(round((0.02 * self.equity) / price))
                if moon_units > 0:
                    self.buy(size=moon_units)
                    print(f"🌕 LUNAR LIFT-OFF! Long {moon_units} units at {price:.2f} | Volume {volume:.2f} > {1.5*volume_avg:.2f}")
            
            # Short Entry: Price breaches lower 2σ with volume surge
            elif price < self.lower_2std[-1] and volume >= 1.5 * volume_avg:
                moon_units = int(round((0.02 * self.equity) / price))
                if moon_units > 0:
                    self.sell(size=moon_units)
                    print(f"🌑 DARK SIDE DESCENT! Short {moon_units} units at {price:.2f} | Volume {volume:.2f} > {1.5*volume_avg:.2f}")
        
        else:
            # Long Exit: Price returns to upper 1σ
            if self.position.is_long and price <= self.upper_1std[-1]:
                self.position.close()
                print(f"🌖 GRAVITY CALLS! Closing long at {price:.2f} | 1σ Band: {self.upper_1std[-1]:.2f}")
            
            # Short Exit: Price returns to lower 1σ
            elif self.position.is_short and price >= self.lower_1std[-1]:
                self.position.close()
                print(f"🌒 LIGHT RETURNS! Covering short at {price:.2f} | 1σ Band: {self.lower_1std[-1]:.2f}")

# Launch Moon Mission 🚀🌙
bt = Backtest(data, VolSurgeBreakout, cash=1_000_000, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._str