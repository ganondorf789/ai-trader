Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np  # Added for numerical operations

# Moon Dev themed data preparation 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names with lunar precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySqueezeBreakout(Strategy):
    def init(self):
        # 🌌 Cosmic Bollinger Bands (20,2)
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # 🌠 Keltner Channel (EMA20 + 1.5*ATR20)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.keltner_upper = self.I(lambda ema, atr: ema + 1.5*atr, self.ema20, self.atr20)
        self.keltner_lower = self.I(lambda ema, atr: ema - 1.5*atr, self.ema20, self.atr20)
        
        # 🌓 20-period High/Low
        self.high_20 = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.low_20 = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # 🌕 Volume average
        self.volume_avg_20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # 🌗 Trend filter
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
        # 🌑 ATR for stops
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
    
    def next(self):
        # Moon-themed debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} - Price: {self.data.Close[-1]} ✨")
            print(f"🌌 Indicators - BB Upper: {self.bb_upper[-1]:.2f}, Keltner Upper: {self.keltner_upper[-1]:.2f}")
        
        if not self.position:
            # Check squeeze condition with lunar precision
            squeeze = (self.bb_upper[-1] < self.keltner_upper[-1]) and (self.bb_lower[-1] > self.keltner_lower[-1])
            
            if squeeze:
                # Long conditions for moon launch 🚀
                long_cond = (self.data.Close[-1] > self.high_20[-1]) and \
                            (self.data.Volume[-1] > 1.5*self.volume_avg_20[-1]) and \
                            (self.data.Close[-1] > self.sma200[-1])
                
                # Short conditions for cosmic re-entry 🌠
                short_cond = (self.data.Close[-1] < self.low_20[-1]) and \
                             (self.data.Volume[-1] > 1.5*self.volume_avg_20[-1]) and \
                             (self.data.Close[-1] < self.sma200[-1])
                
                if long_cond:
                    risk_amount = self