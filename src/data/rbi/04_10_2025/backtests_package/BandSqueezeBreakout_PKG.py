Here's the fixed code with proper Moon Dev themed debug prints and ensuring no backtesting.lib imports are used:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Strategy
from backtesting import Backtest

# Load and prepare data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data[['open', 'high', 'low', 'close', 'volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

class BandSqueezeBreakout(Strategy):
    risk_percent = 0.01
    max_positions = 5
    
    def init(self):
        # Bollinger Bands
        self.upper_bb = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='Upper BB', which=0)
        self.lower_bb = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='Lower BB', which=2)
        
        # Keltner Channel
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_kc = self.I(lambda e,a: e + 1.5*a, self.ema_20, self.atr_20)
        self.lower_kc = self.I(lambda e,a: e - 1.5*a, self.ema_20, self.atr_20)
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # CVI Indicator
        def compute_cvi(h, l, length):
            return ta.cvi(h, l, length=length)
        self.cvi = self.I(compute_cvi, self.data.High, self.data.Low, 9)
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # Squeeze Condition
        self.squeeze = self.I(lambda ub,uk,lb,lk: (ub < uk) & (lb > lk),
                            self.upper_bb, self.upper_kc, self.lower_bb, self.lower_kc)
        
        print("🌙 MOON DEV INIT: Strategy engines primed for launch! 🚀✨")
        print("🌙 INDICATOR STATUS:")
        print(f"   - Bollinger Bands: ONLINE (20,2,2)")
        print(f"   - Keltner Channel: ONLINE (EMA20, ATR20)")
        print(f"   - Volume SMA: ONLINE (20)")
        print(f"   - CVI: ONLINE (9)")
        print(f"   - RSI: ONLINE (14)")
        print("🌙 All systems nominal! Ready for lunar trading! 🌕")

    def next(self):
        # Skip early bars
        if len(self.data) < 24 or len(self.positions) >= self.max_positions:
            return
        
        # Entry signal tracking
        if hasattr(self, 'pending_entry'):
            entry_price = self.data.Open[-1]
            if self.pending_entry['direction'] == 'LONG':
                risk = entry_price - self.pending_entry['sl']
                size = int(round((self.equity * self.risk_percent) / risk))
                if size > 0:
                    self.buy(size=size, sl=self.pending_entry['sl'], 
                            tp=entry_price + 2*risk)
                    print(f"🌙✨ MOON ALERT: Long entered at {entry_price}!")
                    print(f"   - Position Size: {size}")
                    print(f"   - Stop Loss: {self.pending_entry['sl']}")
                    print(f"   - Take Profit: {entry_price + 2*risk}")
                del self.pending_entry
            elif self.pending_entry['direction'] == 'SHORT':
                risk = self.pending_entry['sl'] - entry_price
                size