I'll fix the code by removing the `backtesting.lib` import and replacing any usage of its functions with pure Python/numpy alternatives. Here's the corrected version:

```python
# 🌙 Moon Dev's VolCompressionDivergence Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# 🚀 Data Preparation Magic 🪄
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🌙 Cleanse and Align Cosmic Data ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolCompressionDivergence(Strategy):
    bb_period = 20
    bb_dev = 2
    pcr_ma_period = 10
    lookback_percentile = 20
    risk_pct = 1
    atr_period = 14
    swing_period = 20

    def init(self):
        # 🌗 Bollinger Bandwidth Calculation
        def bbw_calc(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return (upper - lower) / middle
        self.bbw = self.I(bbw_calc, self.data.Close)
        
        # 📉 PCR Smoothing
        self.pcr_ma = self.I(talib.SMA, self.data.PCR, timeperiod=self.pcr_ma_period)
        
        # � BBW Percentile Rollercoaster
        def calc_percentile(series, p):
            return series.rolling(self.lookback_period).apply(
                lambda x: np.percentile(x, p), raw=True
            )
        self.bbw_low = self.I(calc_percentile, self.bbw, 10)
        self.bbw_high = self.I(calc_percentile, self.bbw, 20)
        
        # 🛡️ Risk Management Tools
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        
        print("🌙 Lunar Strategy Activated! 🚀 BBW Engines Ready 📊 PCR Scanners Online 🔍")

    def next(self):
        # 🌑 Current Cosmic Alignment
        price = self.data.Close[-1]
        pcr = self.data.PCR[-1]
        pcr_ma = self.pcr_ma[-1]
        
        if self.position:
            # 🌗 Exit Conditions
            if (self.bbw[-1] > self.bbw_high[-1]) or (pcr < pcr_ma):
                self.position.close()
                print(f"🌙 Moon Exit! 🌗 Price: {price:.2f}")
                print(f"   {'BBW Expanded' if self.bbw[-1] > self.bbw_high[-1] else 'PCR Reversed'} 📉")
        else:
            # 🌕 Entry Conditions
            bbw_compressed = self.bbw[-1] < self.bbw_low[-1]
            pcr_spike = pcr > 1.5 * pcr_ma
            
            if bbw_compressed and pcr_spike:
                # 🎯 Risk Calculation
                risk_amount = self.equity * self.risk_pct / 100
                sl_level = self.swing_high[-1]
                risk_per_share = sl_level - price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.sell(size=size, sl=sl_level)
                        print(f"🌙 SHORT LAUNCH! 🚀 Size: {size}")
                        print(f"   Entry: {price:.2f} 🌑 SL: {sl_level:.2f} 🛡️")

        # 🌓 Cosmic Debugging
        if len(self.data) % 100 == 0:
            print(f"🌙 Bar {len(self.data)} | Price: