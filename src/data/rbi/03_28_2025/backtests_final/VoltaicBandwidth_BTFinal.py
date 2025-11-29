I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicBandwidth(Strategy):
    def init(self):
        # Calculate indicators
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        
        # Dynamic VWAP Bands
        cumulative_tpv = typical_price * self.data.Volume
        rolling_tpv = cumulative_tpv.rolling(20).sum()
        rolling_vol = self.data.Volume.rolling(20).sum()
        vwap = rolling_tpv / rolling_vol
        self.vwap = self.I(lambda: vwap, name='VWAP')
        
        std_dev = talib.STDDEV(self.data.Close, 20)
        self.upper_band = self.I(lambda: vwap + 1.5*std_dev, name='UPPER')
        self.lower_band = self.I(lambda: vwap - 1.5*std_dev, name='LOWER')
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.data.Close, 20, 2, 2)
        self.bb_width = self.I(lambda: (bb_upper - bb_lower)/bb_middle, name='BB_WIDTH')
        
        # Keltner Channels
        atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 20)
        kc_middle = talib.EMA(typical_price, 20)
        kc_upper = kc_middle + 1.5*atr
        kc_lower = kc_middle - 1.5*atr
        self.kc_width = self.I(lambda: kc_upper - kc_lower, name='KC_WIDTH')
        self.kc_sma = self.I(lambda: talib.SMA(self.kc_width, 10), name='KC_SMA')
        
        # Volume MA
        self.vol_ma = self.I(lambda: talib.SMA(self.data.Volume, 20), name='VOL_MA')
        
        print("🌙✨ Moon Dev systems engaged! Trading at lightspeed 🚀")

    def next(self):
        # Moon Dev position tracking
        current_bar = len(self.data) - 1
        
        # Risk management checks
        if len(self.positions) >= 5:
            print("🌑 Maximum positions reached! Holding pattern initiated")
            return
            
        # Entry logic
        price = self.data.Close[-1]
        vol_ok = self.data.Volume[-1] > self.vol_ma[-1]
        bb_expanding = self.bb_width[-1] > self.bb_width[-2] if len(self.bb_width) > 2 else False
        
        # Long entry
        if price > self.upper_band[-1] and bb_expanding and vol_ok:
            sl = self.lower_band[-1]
            risk = price - sl
            if risk <= 0: 
                print("🌘 Invalid risk calculation - aborting trade")
                return
            
            size = int(round((0.01 * self.equity) / risk))
            if size > 0:
                self.buy(size=size, sl=sl, tp=price + 2*risk)
                print(f"🚀🌙 LIFT OFF! Long {size} units @ {price:.2f} | Stop: {sl:.2f}")
        
        # Short entry
        elif price < self.lower_band[-1] and bb_expanding and vol_ok:
            sl = self.upper_band[-1]
            risk = sl - price
            if risk <= 0:
                print("🌘 Invalid risk calculation - aborting trade")
                return
            
            size = int(round((0.01 * self.equity) / risk))
            if size > 0:
                self.sell