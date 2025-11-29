Here's the complete backtesting implementation for the VoltaicSqueeze strategy:

```python
# 🌙 Moon Dev's Voltaic Squeeze Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation magic ✨
def prepare_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    
    # Cleanse and align cosmic energies 🌌
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Universal column alignment 🪐
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

# Cosmic path to enlightenment 📡
DATA_PATH = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = prepare_data(DATA_PATH)

class VoltaicSqueeze(Strategy):
    risk_percentage = 0.01  # 1% cosmic energy per trade 🌠
    
    def init(self):
        # Stellar indicator configuration 🌟
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Quantum Bollinger Bands 🌗
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                    timeperiod=20,
                                                    nbdevup=2,
                                                    nbdevdn=2,
                                                    matype=0)
        self.bb_upper = self.I(lambda: bb_upper, name='BB_Upper')
        self.bb_middle = self.I(lambda: bb_middle, name='BB_Middle')
        self.bb_lower = self.I(lambda: bb_lower, name='BB_Lower')
        
        # Gravitational Squeeze Detection 🌘
        bb_width = (bb_upper - bb_lower) / bb_middle
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')
        self.min_bb_width = self.I(talib.MIN, self.bb_width, timeperiod=30, name='Min_BB_Width')
        
        # Cosmic Volatility Measure 🌪️
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14, name='ATR')
        
        # Temporal position tracking ⏳
        self.entry_bar = 0

    def next(self):
        # Cosmic debug output 🌐
        current_close = self.data.Close[-1]
        print(f"🌙 Lunar Pulse Check >> Close: {current_close:.2f} | "
              f"BB Width: {self.bb_width[-1]:.5f} | "
              f"ATR: {self.atr[-1]:.2f} ✨")
        
        if self.position:
            # Active constellation monitoring 🌠
            print(f"🚀 Warp Drive Engaged >> {self.position.type} "
                  f"Size: {self.position.size} "
                  f"Entry: {self.position.entry_price:.2f} 🌌")
            
            # Exit protocol activation sequence 🔚
            if self.position.is_long:
                if self._exit_long_conditions():
                    self.position.close()
            else:
                if self._exit_short_conditions():
                    self.position.close()
        else:
            # Squeeze detection matrix activated 🔍
            if self.bb_width[-1] <= self.min_bb_width[-1]:
                if current_close > self.bb_upper[-1]:
                    self._enter_long()
                elif current_close < self.bb_lower[-1]:
                    self._enter_short()

    def _enter_long(self):
        # Stellar position calculation 🌟
        stop_distance = 1.5 * self.atr[-1]
        position_size = self._calculate_position_size(stop_distance)
        
        if position_size > 0:
            sl = self.data.Close[-1] - stop_distance
            self.buy(size=position_size, sl=sl)
            self.entry_bar = len(self.data)
            print(f"🚀🌖 LONG TRANSMISSION INITIATED >> "
                  f"Size: {position_size} "
                  f"Entry: {self.data.Close[-1]:.2f} "
                  f"SL: {sl:.2f} ✅")

    def _enter_short(self):
        # Anti-matter position calculation 🌑
        stop_distance = 1.5