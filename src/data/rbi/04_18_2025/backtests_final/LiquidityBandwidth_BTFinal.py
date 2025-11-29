Here's the fixed backtest code with Moon Dev themed debug prints and proper technical implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preprocessing with Moon Dev validation
print("🌙 MOON DEV DATA PREP | Loading celestial market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping with lunar precision
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set index with cosmic alignment
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("🌌 DATA ALIGNMENT COMPLETE | Temporal coordinates synchronized")

class LiquidityBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_daily_loss = 0.05  # 5% max daily loss
    consecutive_losses = 0
    
    def init(self):
        # Bollinger Bands initialization
        print("🌕 INITIALIZING BOLLINGER BANDS | Calculating celestial boundaries...")
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
        
        # Bollinger Bandwidth calculations
        print("🌓 CALCULATING BANDWIDTH OSCILLATIONS | Measuring market compression...")
        self.bw = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle*100, name='BW')
        self.bw_min = self.I(talib.MIN, self.bw, 19200, name='BW_MIN')
        self.bw_max = self.I(talib.MAX, self.bw, 19200, name='BW_MAX')
        self.bw_pct = self.I(lambda: (self.bw-self.bw_min)/(self.bw_max-self.bw_min)*100, name='BW_PCT')
        
        # Funding Rate Z-Score
        print("🌗 ANALYZING FUNDING TIDES | Calculating z-scores...")
        fr = self.data['funding_rate']
        self.fr_sma = self.I(talib.SMA, fr, 30, name='FR_SMA')
        self.fr_std = self.I(talib.STDDEV, fr, 30, name='FR_STD')
        self.fr_z = self.I(lambda: (fr-self.fr_sma)/self.fr_std, name='FR_Z')
        
        # Volume SMA
        print("🌔 MEASURING VOLUME PULSES | Smoothing volume data...")
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOL_SMA')
        
        # ATR for risk calculation
        print("🌑 CALCULATING VOLATILITY | Determining true price ranges...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
    def next(self):
        # Skip if 3 consecutive losses
        if self.consecutive_losses >= 3:
            print("🌙 MOON DEV LOSS BREAKER 🚫 | 3 consecutive losses - skipping trades")
            return
            
        # Long entry conditions
        if (self.bw_pct[-1] <= 5 and
            self.fr_z[-1] <= -2 and
            self.data.Close[-1] >