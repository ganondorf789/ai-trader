```python
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import numpy as np

class VolatilitySqueeze(Strategy):
    # Strategy parameters
    bb_timeperiod = 20
    bb_devup = 2
    bb_devdn = 2
    funding_rate_threshold = 0.001  # 0.1%
    squeeze_threshold = 0.05        # 5% bandwidth
    vw_lookback = 20
    
    def init(self):
        # Precompute all indicators
        close = self.data.Close.values
        high = self.data.High.values
        low = self.data.Low.values
        volume = self.data.Volume.values
        
        # Bollinger Bands calculation
        self.upper_bb, self.middle_bb, self.lower_bb = talib.BBANDS(
            close, 
            timeperiod=self.bb_timeperiod,
            nbdevup=self.bb_devup,
            nbdevdn=self.bb_devdn
        )
        
        # Volume-weighted price extremes
        high_volume = high * volume
        low_volume = low * volume
        
        sum_high_volume = talib.SUM(high_volume, self.vw_lookback)
        sum_low_volume = talib.SUM(low_volume, self.vw_lookback)
        sum_volume = talib.SUM(volume, self.vw_lookback)
        
        self.vw_high = sum_high_volume / sum_volume
        self.vw_low = sum_low_volume / sum_volume
        
        # Add indicators using self.I()
        self.I(lambda: self.upper_bb, name='UpperBB')
        self.I(lambda: self.middle_bb, name='MiddleBB')
        self.I(lambda: self.lower_bb, name='LowerBB')
        self.I(lambda: self.vw_high, name='VW_High')
        self.I(lambda: self.vw_low, name='VW_Low')
        
        print("🌕 Moon Dev Indicators Activated! BB + VWAP extremes loaded")

    def next(self):
        # Skip if position exists or insufficient data
        if self.position or len(self.data) < self.vw_lookback:
            return
            
        current_close = self.data.Close[-1]
        current_upper = self.upper_bb[-1]
        current_lower = self.lower_bb[-1]
        current_middle = self.middle_bb[-1]
        current_funding = self.data.Funding_Rate[-1]
        current_vw_low = self.vw_low[-1]
        
        # Calculate Bollinger Band squeeze
        bandwidth = (current_upper - current_lower) / current_middle
        squeeze = bandwidth < self.squeeze_threshold
        
        # Entry conditions
        funding_condition = current_funding > self.funding_rate_threshold
        price_condition = current_close < current_lower
        
        if squeeze and funding_condition and price_condition:
            # Risk management calculations
            equity = self.equity
            risk_percent = 0.01
            entry_price = current_close
            stop_loss = current_upper
            
            risk_per_unit = abs(stop_loss - entry_price)
            if risk_per_unit == 0:
                print("🌑 Zero risk detected! Aborting launch sequence")
                return
                
            position_size = (equity * risk_percent) / risk_per_unit
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.sell(
                    size=position_size,
                    sl=stop_loss,
                    tp=current_vw_low,
                    tag="Moon_Short"
                )
                print(f"🚀🌒 MOON DEV SHORT! Size: {position_size}")
                print(f"   Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {current_vw_low:.2f}")

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '')
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Find and rename funding rate column
funding_col = [col for col in data.columns if 'funding' in col][0]
data.rename(columns={funding_col: 'Funding_Rate'}, inplace=True)

# Convert and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index