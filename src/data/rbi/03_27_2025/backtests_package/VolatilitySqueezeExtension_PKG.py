Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeExtension(Strategy):
    bb_period = 20
    bb_std = 2
    chaikin_fast = 3
    chaikin_slow = 10
    risk_pct = 0.01
    fib127 = 1.27
    fib1618 = 1.618

    def init(self):
        # Bollinger Bands
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[0], 
                              self.data.Close, name='UpperBB')
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[2], 
                              self.data.Close, name='LowerBB')
        
        # Bollinger Band Width
        self.bb_width = self.I(lambda: (self.upper_bb - self.lower_bb) / 
                              talib.SMA(self.data.Close, self.bb_period), name='BB_Width')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, self.bb_period, name='BB_Width_Avg')
        
        # Chaikin Oscillator
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, self.data.Volume,
                             self.chaikin_fast, self.chaikin_slow, name='Chaikin')
        
        # Volatility measures
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.atr_sma = self.I(talib.SMA, self.atr, 20, name='ATR_SMA')
        
        # Swing lows
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')

    def next(self):
        if self.position:
            return

        # Check entry conditions within 3-bar window
        window = 3
        valid_squeeze = any(self.bb_width[-i] < 0.3*self.bb_width_avg[-i] for i in range(1, window+1))
        
        # Moon Dev replacement for crossover
        valid_chaikin = any((self.chaikin[-i-2] < 0 and self.chaikin[-i-1] > 0) for i in range(1, window+1))
        
        valid_breakout = any(self.data.Close[-i] > self.upper_bb[-i] for i in range(1, window+1))

        if valid_squeeze and valid_chaikin and valid_breakout:
            # Calculate Fibonacci levels
            squeeze_high = max(self.data.High[-20:])
            squeeze_low = min(self.data.Low[-20:])
            fib_range = squeeze_high - squeeze_low
            
            entry_price = self.data.Close[-1]
            target1 = entry_price + self.fib127 * fib_range
            target2 = entry_price + self.fib1618 * fib_range
            
            # Determine stop loss
            stop_price = min(self.swing_low[-1], self.lower_bb[-1])
            
            # Calculate position size with volatility adjustment
            risk_amount = self.risk_pct * self.equity
            risk_per_share = entry_price - stop_price
            position_size = risk_amount / risk_per_share
            
            if self.atr[-1] > self.atr_sma[-1] * 1.5:
                position_size *= 0.5
                
            position_size = int(round(position_size))
            if position_size <= 0:
                return

            # Enter