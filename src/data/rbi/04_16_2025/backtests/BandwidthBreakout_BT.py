```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandwidthBreakout(Strategy):
    initial_capital = 1_000_000
    
    def init(self):
        # 🌙 MOON DEV INDICATOR SETUP START
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_Lower')
        
        # Bollinger Width calculations
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB_Width')
        
        # Historical percentile calculation
        def bb_percentile(series):
            return series.rolling(20).quantile(0.2)
        self.bb_pct = self.I(bb_percentile, self.bb_width, name='BB_Percentile')
        
        # Keltner Channels (using pandas_ta)
        def kc_upper(high, low, close):
            kc = pandas_ta.kc(high, low, close, length=20, scalar=1.5)
            return kc.iloc[:,0]
        self.kc_upper = self.I(kc_upper, self.data.High, self.data.Low, self.data.Close, name='KC_Upper')
        
        def kc_lower(high, low, close):
            kc = pandas_ta.kc(high, low, close, length=20, scalar=1.5)
            return kc.iloc[:,2]
        self.kc_lower = self.I(kc_lower, self.data.High, self.data.Low, self.data.Close, name='KC_Lower')
        
        # Swing High/Low for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        # 🌙 MOON DEV INDICATOR SETUP COMPLETE
        
    def next(self):
        price = self.data.Close[-1]
        
        # 🌙✨ MOON DEV SIGNAL CHECK
        if len(self.data) < 20:  # Warmup period
            return
            
        # Current indicator values
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        kc_upper = self.kc_upper[-1]
        kc_lower = self.kc_lower[-1]
        bb_width = self.bb_width[-1]
        bb_pct = self.bb_pct[-1]
        
        # Keltner expansion check
        kc_expanding = (self.kc_upper[-1] > self.kc_upper[-2] and 
                       self.kc_lower[-1] < self.kc_lower[-2])
        
        # 🚀 ENTRY LOGIC
        if not self.position:
            # Long entry condition
            if (price > bb_upper and price > kc_upper and 
                bb_width < bb_pct and kc_expanding):
                
                sl = self.swing_low[-1]
                risk = price - sl
                if risk <= 0:
                    return
                
                # 🌕 MOON DEV RISK CALCULATION
                risk_amount = self.equ