Here's the fully debugged and fixed backtest code for the LiquidationBandwidth strategy with Moon Dev themed improvements:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR LIQUIDATIONBANDWIDTH STRATEGY 🚀
# 🚀 COMPLETELY FREE FROM backtesting.lib DEPENDENCIES 🌕

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========================
# DATA PREPROCESSING
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# STRATEGY IMPLEMENTATION
# ========================
class LiquidationBandwidthStrategy(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    bb_period = 20
    swing_period = 50
    volatility_threshold = 1.5  # 1.5x average BB width for exit
    
    def init(self):
        # 🌙 CORE INDICATORS (PURE TALIB IMPLEMENTATION)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOLUME_SMA')
        
        # 🌙 BOLLINGER BAND COMPONENTS (NO BACKTESTING.LIB USAGE)
        self.sma = self.I(talib.SMA, self.data.Close, self.bb_period, name='SMA20')
        self.std = self.I(talib.STDDEV, self.data.Close, self.bb_period, name='STD20')
        self.upper = self.I(lambda sma, std: sma + 2*std, self.sma, self.std, name='UPPER_BB')
        self.lower = self.I(lambda sma, std: sma - 2*std, self.sma, self.std, name='LOWER_BB')
        self.bb_width = self.I(lambda u, l, s: (u - l)/s, self.upper, self.lower, self.sma, name='BB_WIDTH')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 10, name='BB_WIDTH_AVG')

        print("🌕 MOON DEV INDICATORS INITIALIZED SUCCESSFULLY! ALL SYSTEMS GO! 🚀")

    def next(self):
        # 🌙 MOON DEV DEBUG DATA
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        bbw = self.bb_width[-1]
        bbw_avg = self.bb_width_avg[-1]
        
        # 🌙 EXIT CONDITIONS (VOLATILITY EXPANSION)
        if self.position and bbw > bbw_avg * self.volatility_threshold:
            self.position.close()
            print(f"🌑 MOON PHASE SHIFT: Volatility Expansion | Closing Position | BB Width: {bbw:.4f} vs Avg: {bbw_avg:.4f}")
            
        if not self.position:
            # 🌙 LONG ENTRY CONDITIONS
            long_trigger = (current_close > self.swing_high[-1] 
                          and current_close > self.upper[-1]
                          and bbw < bbw_avg
                          and current_volume > self.volume_sma[-1])
            
            if long_trigger:
                sl_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(current_close - sl_price)
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, tag='MO