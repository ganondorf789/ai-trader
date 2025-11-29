Here's the fixed code with all indicator packages properly implemented and Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import pandas_ta
import numpy as np

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

class VolSurgeBand(Strategy):
    bb_period = 20
    bb_dev = 2
    bbw_percentile_length = 100
    volume_sma_length = 20
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands components
        def get_bb_upper(close, timeperiod, nbdev):
            upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
            return upper
            
        def get_bb_middle(close, timeperiod, nbdev):
            _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
            return middle
            
        def get_bb_lower(close, timeperiod, nbdev):
            _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
            return lower

        self.bb_upper = self.I(get_bb_upper, self.data.Close, self.bb_period, self.bb_dev, name='BB_UPPER')
        self.bb_middle = self.I(get_bb_middle, self.data.Close, self.bb_period, self.bb_dev, name='BB_MIDDLE')
        self.bb_lower = self.I(get_bb_lower, self.data.Close, self.bb_period, self.bb_dev, name='BB_LOWER')
        
        # Bollinger Bandwidth
        self.bbw = self.I(lambda u, l, m: (u - l) / m, 
                         self.bb_upper, self.bb_lower, self.bb_middle, name='BBW')
        
        # BBW Percentile Rank using pandas_ta
        self.bbw_percentile = self.I(pandas_ta.percentrank, self.bbw, length=self.bbw_percentile_length, name='BBW_PCT')
        
        # Volume indicators using talib
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_sma_length, name='VOL_SMA20')
        
    def next(self):
        # Moon-themed debug logging
        if len(self.data) > 1:
            debug_str = (
                f"🌙 MOON DEV DEBUG 🌙\n"
                f"| Close: {self.data.Close[-1]:.2f} | BBW%: {self.bbw_percentile[-1]:.1f} |\n"
                f"| Volume: {self.data.Volume[-1]:.0f} vs {self.volume_sma20[-1]:.0f} |"
            )
            print(debug_str)
        
        # Exit conditions
        if self.position.is_long and self.data.High[-1] >= self.bb_upper[-1]:
            print(f"🚀🌕 PROFIT TAKEN! Price touched Upper Band @ {self.bb_upper[-1]:.2f}")
            self.position.close()
            
        if self.position.is_short and self.data.Low[-1] <= self.bb_lower[-1]:
            print(f"🌑🚀 COVER SHORT! Price touched Lower Band @ {self.bb_lower[-1]:.2f}")
            self.position.close()
            
        # Entry conditions
        if not self.position:
            current_volume = self.data.Volume[-1]
            vol_ratio = current_volume / self.volume_sma20[-1]