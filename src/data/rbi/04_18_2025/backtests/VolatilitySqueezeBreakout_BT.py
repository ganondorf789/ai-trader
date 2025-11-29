```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas_ta as ta

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Set datetime index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class VolatilitySqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # Calculate indicators using TA-Lib 🌟
        close = self.data.Close
        volume = self.data.Volume
        
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0),
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Bollinger Band Width
        bb_width = self.bb_upper - self.bb_lower
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')
        
        # BB Width Percentile (20th percentile over 100 periods)
        self.bb_width_percentile = self.I(
            lambda: ta.quantile(bb_width, length=100, q=0.2),
            name='BB_Width_Percentile'
        )
        
        # Volume MA (20 periods)
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=20, name='Volume_MA')
        
        # Initialize consolidation tracking variables 🌌
        self.in_consolidation = False
        self.consolidation_start = None
        self.cumulative_pv = 0
        self.cumulative_vol = 0
        self.closes_in_consolidation = []

    def next(self):
        # Skip first 100 bars for indicator warmup 🌙
        if len(self.data) < 100:
            return

        # Update consolidation tracker 🔄
        current_bb_width = self.bb_width[-1]
        current_percentile = self.bb_width_percentile[-1]
        
        if current_bb_width < current_percentile:
            if not self.in_consolidation:
                # New consolidation detected 🌟
                self.in_consolidation = True
                self.consolidation_start = len(self.data) - 1
                self.cumulative_pv = 0
                self.cumulative_vol = 0
                self.closes_in_consolidation = []
                print(f"🌙 MOON DEV ALERT 🌙 | New consolidation started at {self.data.index[-1]}")
            
            # Update cumulative values for VWAP calculation 🧮
            typical_price = (self.data.High[-1] + self.data.Low[-1] + self.data.Close[-1]) / 3
            self.cumulative_pv += typical_price * self.data.Volume[-1]
            self.cumulative_vol += self.data.Volume[-1]
            self.closes_in_consolidation.append(self.data.Close[-1])
        else:
            self.in_consolidation = False

        # Check entry conditions only if not in position 🚀
        if not self.position:
            volume_condition = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
            
            # Long Entry 🌕
            if (self.data.Close[-1] > self.bb_upper[-1] and 
                volume_condition and 
                self.in_consolidation):
                
                self._enter_trade(direction='long')
            
            # Short Entry 🌑
            elif (self.data.Close[-1] < self.bb_lower[-1] and 
                  volume_condition and 
                  self.in_consolidation):
                
                self._enter_trade(direction='short')

    def _enter_trade(self, direction):
        # Risk management calculations 🔒
        entry_price = self.data.Close