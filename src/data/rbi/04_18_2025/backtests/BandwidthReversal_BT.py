```python
# 🌙 Moon Dev Backtest AI Implementation
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌌 Cosmic Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Data Cleaning Ritual
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class BandwidthReversal(Strategy):
    def init(self):
        # 🌠 Stellar Indicators Configuration
        # Bollinger Bands with TA-Lib
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 🌗 Bandwidth Calculation
        self.bandwidth = (self.bb_upper - self.bb_lower) / self.bb_middle
        
        # 📊 Percentile Calculations using Rolling Magic
        self.bb_width_10 = self.I(lambda x: x.rolling(100).apply(
            lambda y: np.percentile(y, 10)), self.bandwidth, name='BB_WIDTH_10')
        self.bb_width_25 = self.I(lambda x: x.rolling(100).apply(
            lambda y: np.percentile(y, 25)), self.bandwidth, name='BB_WIDTH_25')
        
        # 🚀 Momentum Indicators
        self.roc3 = self.I(talib.ROC, self.data.Close, timeperiod=3)
        self.roc10 = self.I(talib.ROC, self.data.Close, timeperiod=10)
        
        # 📈 Volume Filter
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # 🛡️ Risk Management Guardians
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
            self.data.Close, timeperiod=14)

    def next(self):
        # 🌑 Skip if not enough cosmic data
        if len(self.data) < 100:
            return

        # 🌠 Current Celestial Readings
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bandwidth = self.bandwidth[-1]
        bbw_10 = self.bb_width_10[-1]
        bbw_25 = self.bb_width_25[-1]
        roc3 = self.roc3[-1]
        roc3_prev = self.roc3[-2]
        roc10 = self.roc10[-1]
        roc10_prev = self.roc10[-2]
        vol_sma20 = self.volume_sma20[-1]
        atr = self.atr[-1]

        # 🌙 Debug Console
        print(f"\n🌕 Lunar Cycle Update 🌕\nClose: {current_close:.2f} | "
              f"BBW: {bandwidth:.4f} | ROC3: {roc3:.2f}% | ROC10: {roc10:.2f}%")

        # 🚀 Trade Entry Protocol
        if not self.position:
            # 🌊 Volume Filter Check
            if current_volume < vol_sma20:
                print("🚫 Volume tsunami too weak - aborting launch")
                return

            # 🌕 Long Entry Sequence
            long_cond = (
                bandwidth < bbw_10 and
                roc3 > roc10 and roc3_prev <= roc10_prev and
                current_close <= bb_lower + 0.3*(bb_upper - bb_lower)
            )
            
            # 🌑 Short Entry Sequence
            short_cond = (
                bandwidth < bbw_10 and
                roc3 < roc10 and roc3_prev >= roc10_prev and
                current_close >= bb_upper - 0.3*(bb_upper - bb_lower)