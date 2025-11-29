Here's the fixed code with all indicators properly implemented without any backtesting.lib usage, along with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert to datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class BandwidthBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate Bollinger Bands using talib
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['Upper', 'Middle', 'Lower']
        )
        
        # Bollinger Band width calculation
        self.bb_width = self.I(lambda u, l: u - l, self.upper, self.lower, name='BB Width')
        
        # 10-period low of BB width using talib MIN
        self.bb_low = self.I(talib.MIN, self.bb_width, timeperiod=10, name='BB Low')
        
        # Volume indicators using talib SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Vol SMA')
        
        # ATR for risk management using talib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=14, name='ATR')
        
        # Previous periods' highs/lows using pandas shift
        self.prior_high = self.I(lambda x: x.shift(1), self.data.High, name='Prior High')
        self.prior_low = self.I(lambda x: x.shift(1), self.data.Low, name='Prior Low')
        
        # Trackers for positions
        self.trailing_high = None
        self.trailing_low = None
        self.take_profit = None

    def next(self):
        current_close = self.data.Close[-1]
        
        if not self.position:
            # Entry conditions using numpy/pandas comparisons
            bb_cond = (self.bb_width[-1] == self.bb_low[-1])
            vol_cond = (self.data.Volume[-1] > self.volume_sma[-1])
            
            # Long entry
            if bb_cond and vol_cond and (self.data.High[-1] > self.prior_high[-1]):
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                stop_distance = 3 * atr_value
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=current_close - stop_distance,
                            tag='Long Entry')
                    self.trailing_high = current_close
                    self.take_profit = current_close + 3 * atr_value
                    print(f"🌙✨ MOON ALERT! LONG SIGNAL DETECTED! ✨🌙")
                    print(f"   Entry: {current_close:.2f} | Size: {position_size} contracts")
                    print(f"   Stop Loss: {current_close - stop_distance:.2f}")
                    print(f"   Take Profit: {self.take_profit:.2f}")
                    print(f"   🚀 Ready for liftoff! 🚀\n")
            
            # Short entry
            elif bb_cond and vol_cond and (self.data.Low[-1] < self.prior_low[-1]):
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                stop_distance = 3 * atr_value
                position_size = int(round(risk_amount / stop_distance))