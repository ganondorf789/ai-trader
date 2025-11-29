```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match required format
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricSqueeze(Strategy):
    def init(self):
        # Calculate indicators
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume
        
        # Klinger Oscillator
        kvo = ta.kvo(high, low, close, volume, fast=34, slow=55, signal=13)
        self.kvo_line = self.I(lambda: kvo.iloc[:,0], name='KVO')
        self.signal_line = self.I(lambda: kvo.iloc[:,1], name='KVO_Signal')
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: bb_upper, name='Upper BB')
        self.bb_middle = self.I(lambda: bb_middle, name='Middle BB')
        self.bb_lower = self.I(lambda: bb_lower, name='Lower BB')
        
        # Bollinger Width
        bb_width = (bb_upper - bb_lower) / bb_middle * 100
        self.bb_width = self.I(lambda: bb_width, name='BB Width')
        
        # BB Width extremes
        self.bbw_max = self.I(talib.MAX, self.bb_width, 100, name='BBW_MAX')
        self.bbw_min = self.I(talib.MIN, self.bb_width, 100, name='BBW_MIN')
        
        # Parabolic SAR
        self.sar = self.I(talib.SAR, high, low, 0.02, 0.2, name='SAR')
        
        # Swing Low
        self.swing_low = self.I(talib.MIN, low, 20, name='Swing Low')
        
        print("🌙 VolumetricSqueeze initialized with Moon Dev power! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        
        if not self.position:
            # Entry conditions
            kvo_cross = self.kvo_line[-2] < self.signal_line[-2] and self.kvo_line[-1] > self.signal_line[-1]
            
            bbw_percentile = (self.bb_width[-1] - self.bbw_min[-1]) / (self.bbw_max[-1] - self.bbw_min[-1] + 1e-9)
            bb_squeeze = bbw_percentile <= 0.3
            
            price_above_bb = current_close > self.bb_middle[-1]
            
            if kvo_cross and bb_squeeze and price_above_bb:
                # Calculate stop loss
                sl_price = min(self.swing_low[-1], self.bb_lower[-1])
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    risk_amount = self.equity * 0.01
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        print(f"🚀 MOON LAUNCH: Long {position_size} @ {current_close:.2f} | SL: {sl_price:.2f} 🌕")
                        self.buy(size=position_size, sl=sl_price)
        else:
            # Exit conditions
            current_size = self.position.size
            
            # SAR exit
            sar_below = self.sar[-1] < current_close and self.sar[-2] >= self.data.Close[-2]
            if sar_below:
                print(f"🌑 SAR Exit @ {current