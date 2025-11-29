Here's the fixed code with all Moon Dev themed improvements and proper indicator package usage:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Moon Dev's Cosmic Data Preparation 🌌
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse the cosmic dust from column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align celestial bodies with proper case constellations 🌠
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set temporal coordinates as index 🕰️
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class SqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% of lunar treasury 🌙
    
    def init(self):
        # Cosmic indicator calculations using TA-Lib's star maps 🌟
        # Bollinger Bands (20, 2.0)
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[2], self.data.Close)
        
        # Keltner Channels (20 EMA + 1.5 ATR)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.kc_upper = self.I(lambda ema, atr: ema + 1.5*atr, self.ema, self.atr)
        self.kc_lower = self.I(lambda ema, atr: ema - 1.5*atr, self.ema, self.atr)
        
        # Volume MA and ADX for trend confirmation 🌪️
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙 Lunar indicators initialized! Ready for cosmic breakout detection 🚀")

    def next(self):
        # Avoid premature launches during cosmic alignment 🌑
        if len(self.data) < 20:
            return

        # Current phase measurements 🌗
        close = self.data.Close[-1]
        squeeze = (self.bb_upper[-1] < self.kc_upper[-1]) and (self.bb_lower[-1] > self.kc_lower[-1])
        vol_ok = self.data.Volume[-1] > self.vol_ma[-1]  # Fixed volume condition (should be > for breakout)
        trend_ok = self.adx[-1] > 20

        # Cosmic entry protocol 🌌
        if not self.position:
            if squeeze and vol_ok and trend_ok:
                # Long entry conditions 🚀
                if close > self.bb_upper[-1] and close > self.kc_upper[-1]:
                    self.enter_long()
                
                # Short entry conditions 🌑
                elif close < self.bb_lower[-1] and close < self.kc_lower[-1]:
                    self.enter_short()
        
        # Galactic exit protocol 💫
        else:
            if self.position.is_long and close < self.ema[-1]:  # Changed to EMA for better exit
                self.position.close()
                print(f"🌗 Closing LONG at {close} - Price crossed below EMA ✨")
            elif self.position.is_short and close > self.ema[-1]:
                self.position.close()
                print(f"🌓 Closing SHORT at {close} - Price crossed above EMA ✨")

    def enter_long(self):
        # Moon gravity risk calculations 🌕
        atr = self.atr[-1]
        entry_price = self.data.Close[-1]
        sl = entry_price - 1.5 * atr
        risk = entry_price - sl
        
        if risk <= 0: