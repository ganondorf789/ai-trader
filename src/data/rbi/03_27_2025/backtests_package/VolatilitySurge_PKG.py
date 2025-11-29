Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev's VolatilitySurge Backtest Implementation 🚀

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Clean and prepare data 🌟
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySurge(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    bb_period = 20
    atr_period = 14
    
    def init(self):
        # Bollinger Bands Calculation ✨
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 
                              timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                              timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', which=2)
        self.bb_mid = self.I(talib.BBANDS, self.data.Close, 
                            timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_MID', which=1)
        
        # BB Width Indicators 🌗
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, 
                              self.bb_upper, self.bb_lower, self.bb_mid)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=20)
        
        # Volume and ATR Indicators 🌊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        self.trail_stop = None
        self.peak_price = None

    def next(self):
        # Moon Dev's Trading Logic 🌙
        if not self.position:
            # Entry Conditions Check 🌟
            bb_contraction = self.bb_width[-1] < 0.5 * self.bb_width_ma[-1]
            vol_spike = self.data.Volume[-1] > 1.5 * self.vol_ma[-1]
            price_breakout = self.data.Close[-1] > self.bb_upper[-1]
            
            if bb_contraction and vol_spike and price_breakout:
                # Risk Management Calculation 💰
                atr_value = self.atr[-1] or 1  # Prevent zero division
                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / (atr_value * 1)  # 1x ATR stop
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.peak_price = self.data.High[-1]
                    print(f"🌙 MOON DEV ALERT: Long Entry at {self.data.Close[-1]:.2f} | Size: {position_size} | ATR Stop: {self.data.Close[-1] - atr_value:.2f} 🚀")
        else:
            # Update Trailing Stop 🌈
            self.peak_price = max(self.peak_price, self.data.High[-1])
            self.trail_stop = self.peak_price - 2 * self.atr[-1]
            
            # Exit Condition 🌌
            if self.data.Low[-1] < self.trail_stop:
                self.position.close()
                print(f"🌙 MOON DEV EXIT: Trailing Stop Triggered at {self.trail_stop:.2f} | Profit: {self.position.pl:.2f} ✨")

# Run Backtest and Print Full Stats