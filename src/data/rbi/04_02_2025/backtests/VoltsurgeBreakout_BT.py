Here's the implementation of the VoltsurgeBreakout strategy with all specified requirements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Helper functions for Bollinger Bands calculations
def bbands_upper(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return upper

def bbands_middle(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return middle

def bbands_lower(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return lower

class VoltsurgeBreakout(Strategy):
    def init(self):
        # Calculate all required indicators using TA-Lib
        self.upper = self.I(bbands_upper, self.data.Close, 20, 2, 2, 0)
        self.middle = self.I(bbands_middle, self.data.Close, 20, 2, 2, 0)
        self.lower = self.I(bbands_lower, self.data.Close, 20, 2, 2, 0)
        
        # Bollinger Bandwidth calculation
        self.bollinger_bandwidth = self.I(lambda: (self.upper - self.lower) / self.middle)
        self.bollinger_bandwidth_low = self.I(talib.MIN, self.bollinger_bandwidth, 10)
        
        # Volume and volatility indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        # Entry tracking variables
        self.long_entry_pending = False
        self.short_entry_pending = False
        self.pending_atr = None

    def next(self):
        # Handle pending long entry
        if self.long_entry_pending and not self.position:
            self.execute_long_entry()
            
        # Handle pending short entry
        if self.short_entry_pending and not self.position:
            self.execute_short_entry()
            
        # Check for new entry signals
        if not self.position:
            self.check_entries()
            
        # Manage open positions
        if self.position:
            self.manage_exits()

    def execute_long_entry(self):
        entry_price = self.data.Open[-1]
        risk_amount = self.equity * 0.01
        position_size = int(round(risk_amount / (1.5 * self.pending_atr)))
        
        if position_size > 0:
            self.buy(size=position_size, tag={
                'max_high': entry_price,
                'entry_index': len(self.data) - 1,
                'entry_time': self.data.index[-1]
            })
            print(f"🚀🌙 LONG ENTRY at {entry_price}, Size: {position_size}")
        
        self.long_entry_pending = False
        self.pending_atr = None

    def execute_short_entry(self):
        entry_price = self.data.Open[-1]
        risk_amount = self.equity * 0.01
        position_size = int(round(risk_amount / (1.5 * self.pending_atr)))
        
        if position_size > 0:
            self.sell(size=position_size, tag={
                'min_low': entry_price,
                'entry_index': len(self.data) - 1,
                'entry_time': self.data.index[-1]
            })
            print(f"🚀🌙 SHORT ENTRY at {entry_price}, Size: {position_size}")
        
        self.short_entry_pending = False
        self.pending_atr = None

    def check_entries(self):
        current = {
            'bandwidth': self.bollinger_bandwidth[-1],
            'bandwidth_low': self.bollinger_bandwidth_low[-1],
            '