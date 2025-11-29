Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
import datetime
import numpy as np

# Data handling
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltsqueezeBreakout(Strategy):
    bb_period = 20
    keltner_period = 20
    atr_multiplier = 2.5
    risk_pct = 0.01
    max_positions = 3
    
    def init(self):
        # Bollinger Bands components
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # Bollinger Band Width calculation
        def bb_width(close):
            upper, _, lower = talib.BBANDS(close, timeperiod=self.bb_period)
            return (upper - lower) / ((upper + lower)/2)
        self.bb_width = self.I(bb_width, self.data.Close)
        self.bb_width_percentile = self.I(ta.percentile, self.bb_width, length=100, q=20)
        
        # Keltner Channels
        def keltner_upper(high, low, close):
            ema = talib.EMA(close, self.keltner_period)
            atr = talib.ATR(high, low, close, self.keltner_period)
            return ema + self.atr_multiplier * atr
        self.keltner_upper = self.I(keltner_upper, self.data.High, self.data.Low, self.data.Close)
        
        def keltner_lower(high, low, close):
            ema = talib.EMA(close, self.keltner_period)
            atr = talib.ATR(high, low, close, self.keltner_period)
            return ema - self.atr_multiplier * atr
        self.keltner_lower = self.I(keltner_lower, self.data.High, self.data.Low, self.data.Close)
        
        # Volume and ATR indicators
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 20)
        self.atr10 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 10)
        self.atr22 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 22)
        
        self.active_trades = []

    def next(self):
        current_time = self.data.index[-1].time()
        # Moon Dev session filter 🌓
        if (current_time < datetime.time(0, 30)) or (current_time >= datetime.time(23, 30)):
            print("🌙 Moon Dev: Outside trading hours - skipping execution")
            return
            
        # Moon Dev position counter 🌙
        if len(self.positions) >= self.max_positions:
            print("🌙 Moon Dev: Max positions reached - skipping new entries")
            return

        # Long entry conditions 🚀
        if (self.bb_width_percentile[-1] < 20 and
            self.data.Close[-1] > self.keltner_upper[-1] and
            self.data.Volume[-1] > 1.5 * self.volume_avg[-1]):
            
            entry_price = self.data.Open[0]
            sl_price = self.keltner_lower[-1]
            risk_per_unit = entry_price - sl_price
            
            if risk_per_unit <= 0:
                print("🌙 Moon Dev: Invalid risk calculation - skipping trade")
                return
                
            # Calculate position size