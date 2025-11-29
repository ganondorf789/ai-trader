```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidityBandwidth(Strategy):
    risk_per_trade = 0.01
    max_daily_loss = 0.05
    consecutive_losses = 0
    
    def init(self):
        # Bollinger Bands
        def bb_upper(close):
            u, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return u
        def bb_middle(close):
            _, m, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return m
        def bb_lower(close):
            _, _, l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return l
            
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_UP')
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_MID')
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_LOW')
        
        # Bollinger Bandwidth
        self.bw = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle*100, name='BW')
        self.bw_min = self.I(talib.MIN, self.bw, 19200, name='BW_MIN')
        self.bw_max = self.I(talib.MAX, self.bw, 19200, name='BW_MAX')
        self.bw_pct = self.I(lambda: (self.bw-self.bw_min)/(self.bw_max-self.bw_min)*100, name='BW_PCT')
        
        # Funding Rate Z-Score
        fr = self.data['funding_rate']
        self.fr_sma = self.I(talib.SMA, fr, 30, name='FR_SMA')
        self.fr_std = self.I(talib.STDDEV, fr, 30, name='FR_STD')
        self.fr_z = self.I(lambda: (fr-self.fr_sma)/self.fr_std, name='FR_Z')
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOL_SMA')
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
    def next(self):
        # Skip if 3 consecutive losses
        if self.consecutive_losses >= 3:
            print("🌙 MOON DEV LOSS BREAKER 🚫 | 3 consecutive losses - skipping trades")
            return
            
        # Long entry conditions
        if (self.bw_pct[-1] <= 5 and
            self.fr_z[-1] <= -2 and
            self.data.Close[-1] > self.bb_upper[-1] and
            self.data.Volume[-1] > self.vol_sma[-1]*1.5):
            
            sl = self.data.Low[-1]
            risk_amount = self.risk_per_trade * self.equity
            risk_per_share = self.data.Close[-1] - sl
            size = int(round(risk_amount / risk_per_share))
            
            tp = self.data.Close[-1] + 1.5*self.atr[-1]
            
            print(f"🌙✨ MOON DEV LONG 🚀 | Entry: {self.data.Close[-1]:.2f}")
            print(f"   SL: {sl:.2f} | TP: {tp:.2f} | Size: {size}")
            self.buy(size=size, sl=