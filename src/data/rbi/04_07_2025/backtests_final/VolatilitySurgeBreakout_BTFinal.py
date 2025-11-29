I'll fix the backtest code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import numpy as np

# Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySurgeBreakout(Strategy):
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        print("🌙 Initializing Moon Dev Trading Strategy ✨")
        
        # Bollinger Bands (20,2)
        close = self.data.Close
        bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: bb_upper, name='BB_UPPER')
        self.bb_lower = self.I(lambda: bb_lower, name='BB_LOWER')
        
        # Keltner Channels (20,1.5)
        high, low, close_kc = self.data.High, self.data.Low, self.data.Close
        kc = ta.kc(high=high, low=low, close=close_kc, length=20, multiplier=1.5)
        self.kc_upper = self.I(lambda: kc['KCU_20_1.5'], name='KC_UPPER')
        self.kc_lower = self.I(lambda: kc['KCL_20_1.5'], name='KC_LOWER')
        
        # 5-period RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=5, name='RSI_5')
        
        # Volume SMA (20-period)
        volume = self.data.Volume
        self.volume_avg = self.I(talib.SMA, volume, timeperiod=20, name='VOLUME_SMA_20')
        
        self.consecutive_losses = 0
        print("🌙 Strategy Indicators Initialized Successfully! ✨")

    def next(self):
        # Skip trades if 3 consecutive losses
        if self.consecutive_losses >= 3:
            print("🌙 Skipping trade - 3 consecutive losses! ✨")
            return
        
        if not self.position:
            # Long Entry Conditions
            squeeze = (self.bb_upper[-1] < self.kc_upper[-1]) and (self.bb_lower[-1] > self.kc_lower[-1])
            long_break = (self.data.Close[-1] > self.bb_upper[-1]) and (self.data.Close[-1] > self.kc_upper[-1])
            vol_surge = self.data.Volume[-1] > 1.5 * self.volume_avg[-1]
            
            if squeeze and long_break and vol_surge:
                sl = self.kc_lower[-1]
                risk = self.data.Close[-1] - sl
                if risk > 0:
                    size = int(round((self.equity * 0.01) / risk))
                    self.buy(size=size, sl=sl)
                    print(f"🚀 MOON BOUND! LONG ENTRY: Size {size} @ {self.data.Close[-1]:.2f} | SL: {sl:.2f} ✨")
            
            # Short Entry Conditions
            short_break = (self.data.Close[-1] < self.bb_lower[-1]) and (self.data.Close[-1] < self.kc_lower[-1])
            
            if squeeze and short_break and vol_surge:
                sl = self.kc_upper[-1]
                risk = sl - self.data.Close[-1]
                if risk > 0:
                    size = int(round((self.equity * 0.01) / risk))
                    self.sell(size=size, sl=sl)
                    print(f"🌑 SHORTING THE MOON! Size {size} @ {self.data.Close[-1]:.2f} | SL: {sl:.2f} ✨")