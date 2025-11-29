Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed enhancements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
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

class SqueezeBreakout(Strategy):
    def init(self):
        # Moon Dev Indicator Factory 🌙✨
        print("🌌 INITIALIZING MOON DEV TRADING ORBITAL PLATFORM 🌌")
        
        self.upper_bb = self.I(self._compute_upper_bb, self.data.Close, 20, 2, name='UPPER_BB')
        self.lower_bb = self.I(self._compute_lower_bb, self.data.Close, 20, 2, name='LOWER_BB')
        self.upper_kc = self.I(self._compute_upper_kc, self.data.High, self.data.Low, self.data.Close, 20, 2, name='UPPER_KC')
        self.lower_kc = self.I(self._compute_lower_kc, self.data.High, self.data.Low, self.data.Close, 20, 2, name='LOWER_KC')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA')
        
        print("🌙 MOON DEV SYSTEM INITIALIZED 🌙\nBollinger Bands & Keltner Channels Armed ✨")
        print("🚀 Ready for Lunar Trading Missions 🚀")

    @staticmethod
    def _compute_upper_bb(close, period, dev):
        sma = talib.SMA(close, timeperiod=period)
        std = talib.STDDEV(close, timeperiod=period)
        return sma + dev * std

    @staticmethod
    def _compute_lower_bb(close, period, dev):
        sma = talib.SMA(close, timeperiod=period)
        std = talib.STDDEV(close, timeperiod=period)
        return sma - dev * std

    @staticmethod
    def _compute_upper_kc(h, l, c, period, mult):
        ema = talib.EMA(c, timeperiod=period)
        atr = talib.ATR(h, l, c, timeperiod=period)
        return ema + mult * atr

    @staticmethod
    def _compute_lower_kc(h, l, c, period, mult):
        ema = talib.EMA(c, timeperiod=period)
        atr = talib.ATR(h, l, c, timeperiod=period)
        return ema - mult * atr

    def next(self):
        # Moon Dev Safety Check 🛡️
        if len(self.data) < 20 or self.data.Volume[-1] == 0:
            print("⚠️ MOON DEV ALERT: Insufficient data or zero volume - skipping cycle")
            return

        price = self.data.Close[-1]
        upper_bb = self.upper_bb[-1]
        lower_bb = self.lower_bb[-1]
        upper_kc = self.upper_kc[-1]
        lower_kc = self.lower_kc[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.volume_ma[-1]

        # Moon Dev Signal Scanner 📡
        if not self.position:
            # Long Entry 🌕
            if (price > upper_bb) and (price < upper_kc) and (vol > vol_ma):
                risk_amount = self.equity * 0.01
                sl = lower_kc
                risk_per_unit = abs(price - sl)
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        self.buy(size=size, sl=sl, tp=price + 2*(price-sl))
                        print(f"🚀 MOON DEV LONG LAUNCH 🚀