I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Liquidation Cluster Breakout Backtest 🌙
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# Data preprocessing moon-style 🌕
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'Date'
    }, inplace=True)
    # Convert and set index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Load cosmic crypto data 🌌
data = load_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

class LiquidationClusterBreakout(Strategy):
    # Strategy parameters 🚀
    atr_period = 20
    swing_window = 20
    volume_window = 20
    risk_pct = 0.01  # 1% risk per trade
    atr_multiplier = 1.0  # 1x ATR for SL
    rr_ratio = 2  # Risk:Reward ratio

    def init(self):
        # Cosmic indicators 🌠
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window, name='SWING LOW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period, name='ATR')
        self.avg_volume = self.I(talib.SMA, self.data.Volume, 
                                timeperiod=self.volume_window, name='AVG VOL')
        
        # Lunar state tracking 🌑🌕
        self.last_swing_high = None
        self.last_swing_low = None
        self.high_volume_retracement = False

    def next(self):
        # Current cosmic readings 🌍
        price = self.data.Close[-1]
        atr_value = self.atr[-1]
        avg_vol = self.avg_volume[-1]
        current_vol = self.data.Volume[-1]

        # Update lunar swing levels 🌗
        if self.data.High[-1] == self.swing_high[-1]:
            self.last_swing_high = self.data.High[-1]
            print(f"🌕 NEW SWING HIGH @ {self.last_swing_high}")
            
        if self.data.Low[-1] == self.swing_low[-1]:
            self.last_swing_low = self.data.Low[-1]
            print(f"🌑 NEW SWING LOW @ {self.last_swing_low}")

        # Only trade when volatility is sleeping 😴
        if atr_value > price * 0.01:  # 1% threshold
            return

        # Long constellation setup 🌟
        if (self.last_swing_high and 
            not self.position and
            price > self.last_swing_high and
            current_vol > avg_vol * 1.5):  # Volume confirmation
            
            # Calculate position size based on risk
            risk_amount = self.equity * self.risk_pct
            stop_loss = price - (atr_value * self.atr_multiplier)
            position_size = risk_amount / (price - stop_loss)
            
            # Round to whole units for crypto trading
            position_size = int(position_size)
            
            if position_size > 0:
                print(f"🚀 LONG SIGNAL: {position_size} units @ {price}")
                self.buy(size=position_size, 
                         sl=stop_loss,
                         tp=price + (self.rr_ratio * (price - stop_loss)))

        # Short constellation setup 🌠
        if (self.last_swing_low and 
            not self.position and
            price < self.last_swing_low and
            current_vol > avg_vol * 1.5):  # Volume