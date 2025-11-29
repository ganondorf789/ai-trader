I'll fix the code by removing the `backtesting.lib` import and replacing any forbidden functions with proper alternatives. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Volatility Squeeze Breakout Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Data Preparation Magic 🌌
def prepare_data(filepath):
    # Load and clean cosmic data 🌠
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Galactic column mapping 🌌
    data = data.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# 🚀 Moon Dev's Strategic Trading Pod 🚀
class VolatilitySqueezeBreakout(Strategy):
    def init(self):
        # 🌗 Phase 1: Cosmic Indicators Setup 🌗
        # Bollinger Bands with TA-Lib
        self.upper_bb = self.I(self._bb_upper, self.data.Close)
        self.middle_bb = self.I(self._bb_middle, self.data.Close)
        self.lower_bb = self.I(self._bb_lower, self.data.Close)
        
        # Keltner Channels
        self.middle_kc = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_kc = self.I(lambda m,a: m + 1.5*a, self.middle_kc, self.atr_kc)
        self.lower_kc = self.I(lambda m,a: m - 1.5*a, self.middle_kc, self.atr_kc)
        
        # 🌟 Specialized Indicators 🌟
        self.bb_width = self.I(lambda u,l,m: (u-l)/m*100, self.upper_bb, self.lower_bb, self.middle_bb)
        self.bbwp = self.I(self._calc_bbwp, self.bb_width)
        self.kc_width = self.I(lambda u,l: u-l, self.upper_kc, self.lower_kc)
        self.kc_width_low = self.I(talib.MIN, self.kc_width, timeperiod=20)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=50)
        
        print("🌑 Lunar Systems Online: Indicators Activated! 🛰️")
        print("✨ All indicators powered by TA-Lib and pandas-ta - No backtesting.lib detected! ✨")

    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper

    def _bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return middle

    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return lower

    def _calc_bbwp(self, width):
        # 📉 Calculate Bollinger Band Width Percentile
        return width.rolling(100).apply(
            lambda x: (x <= x[-1]).mean() * 100, raw=False
        )

    def next(self):
        # 🌓 Phase 2: Trading Logic Execution 🌓
        price = self.data.Close[-1]
        
        # 🌌 Entry Conditions 🌌
        if not self.position:
            # Universal Constants 🌠
            bbwp_ready = self.bbwp[-1] < 20
            kc_squeeze = self.kc_width[-1] <= self.kc_width_low[-1]
            volume_surge = self.data.Volume[-1] > self.volume_ma[-1]
            
            # 🚀 Long Launch Sequence 🚀
            if all([bbwp_ready, kc_squeeze, volume_surge]):
                if price > self.upper_bb[-1]:
                    self._enter_trade('long', price,