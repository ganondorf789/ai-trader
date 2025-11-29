I'll debug the code while maintaining the original strategy logic and adding Moon Dev themed debug prints. Here's the fixed version:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
import pandas_ta as ta

# Data handling
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert and resample to 4H
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data.resample('4H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Convert to US/Eastern time
data.index = data.index.tz_localize('UTC').tz_convert('US/Eastern')

class LiquidationReversion(Strategy):
    initial_equity = 1_000_000
    risk_per_trade = 0.02  # 2% risk per trade
    
    def init(self):
        # Moon Dev indicator initialization 🌙📊
        print("🌙 Initializing Moon Dev indicators...")
        
        # Calculate required indicators with self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_SMA')
        
        # Body size calculations
        self.body_size = self.I(
            lambda close, open: np.abs(close - open), 
            self.data.Close, self.data.Open,
            name='Body_Size'
        )
        self.avg_body_size = self.I(talib.SMA, self.body_size, timeperiod=20, name='Avg_Body_Size')
        
        # Wick calculations
        self.upper_wick = self.I(
            lambda high, open, close: np.where(
                close > open,
                high - close,
                high - open
            ),
            self.data.High, self.data.Open, self.data.Close,
            name='Upper_Wick'
        )
        
        self.lower_wick = self.I(
            lambda low, open, close: np.where(
                close > open,
                open - low,
                close - low
            ),
            self.data.Low, self.data.Open, self.data.Close,
            name='Lower_Wick'
        )
        
        print("🌙 Moon Dev indicators ready for launch! 🚀")
        
    def next(self):
        # Moon Dev time filter 🌙⏰
        current_time = self.data.index[-1].time()
        if not (8 <= current_time.hour < 12):
            return
            
        if len(self.data) < 25:
            return
            
        prev_idx = -2
        current_idx = -1
        
        # Liquidation event checks
        prev_upper_wick = self.upper_wick[prev_idx]
        prev_lower_wick = self.lower_wick[prev_idx]
        prev_avg_body = self.avg_body_size[prev_idx]
        prev_volume = self.data.Volume[prev_idx]
        prev_volume_sma = self.volume_sma[prev_idx]
        
        extreme_wick = (prev_upper_wick >= 3*prev_avg_body) or (prev_lower_wick >= 3*prev_avg_body)
        volume_spike = prev_volume >= 2*prev_volume_sma
        
        if extreme_wick and volume_spike:
            # Moon Dev debug print 🌙🔍
            print(f"🌙 MOON ALERT! Liquidation event detected at {self.data.index[prev_idx]}")
            
            # Bollinger Band touch check
            prev_high = self.data.H