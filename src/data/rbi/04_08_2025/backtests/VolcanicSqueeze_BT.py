```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '')
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to proper case
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolcanicSqueeze(Strategy):
    ema_period = 20
    atr_period = 14
    keltner_multiplier = 2
    volume_multiplier = 1.5
    risk_pct = 0.01
    fib_levels = [1.618, 2.618]
    trailing_atr_mult = 1.5

    def init(self):
        # Core indicators
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, self.ema_period)
        self.heatmap_sma = self.I(talib.SMA, self.data['liquidationheatmap'], self.ema_period)
        
        # Swing extremes
        self.swing_high = self.I(talib.MAX, self.data.High, self.ema_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.ema_period)

    def next(self):
        if self.position:
            self.manage_open_trades()
        else:
            self.check_entries()

    def check_entries(self):
        current_close = self.data.Close[-1]
        upper_band = self.ema[-1] + self.keltner_multiplier * self.atr[-1]
        lower_band = self.ema[-1] - self.keltner_multiplier * self.atr[-1]

        # Volume and heatmap validation
        vol_ok = self.data.Volume[-1] > self.volume_multiplier * self.vol_sma[-1]
        heat_ok = self.data['liquidationheatmap'][-1] > self.volume_multiplier * self.heatmap_sma[-1]

        if current_close > upper_band and vol_ok and heat_ok:
            self.enter_long(upper_band)
        elif current_close < lower_band and vol_ok and heat_ok:
            self.enter_short(lower_band)

    def enter_long(self, upper_band):
        stop_price = self.ema[-1] - self.keltner_multiplier * self.atr[-1]
        risk_distance = self.data.Close[-1] - stop_price
        position_size = self.calculate_size(risk_distance)
        
        if position_size > 0:
            print(f"🚀🌋 LONG ERUPTION! Entry: {self.data.Close[-1]:.2f}")
            self.buy(size=position_size, sl=stop_price, 
                    tag={'atr_entry': self.atr[-1], 'direction': 'long'})

    def enter_short(self, lower_band):
        stop_price = self.ema[-1] + self.keltner_multiplier * self.atr[-1]
        risk_distance = stop_price - self.data.Close[-1]
        position_size = self.calculate_size(risk_distance)
        
        if position_size > 0:
            print(f"🌑🌋 SHORT IMPLOSION! Entry: {self.data.Close[-1]:.2f}")
            self.sell(size=position_size, sl=stop_price,
                     tag={'atr_entry': self.atr[-1], 'direction': 'short'})

    def calculate_size(self, risk_distance):
        if risk_distance <= 0: return 0
        risk_amount = self.risk_pct * self.equity
        size = risk_amount / risk_distance
        return int(round(size))

    def manage_open_trades(self):