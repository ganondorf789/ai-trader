```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR HARMONIC VOLATILITY STRATEGY 🌙

import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# 🌌 DATA PREPARATION WITH PROPER COLUMN MAPPING 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class HarmonicVolatility(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_period = 14
    swing_window = 5
    volatility_multiplier = 2.0
    
    def init(self):
        # 🌗 CORE INDICATORS USING TA-LIB 🌗
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_highs = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_lows = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        
        # 🌓 VOLATILITY BANDS CALCULATION 🌓
        self.midline = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.upper_band = self.I(lambda: self.midline + self.volatility_multiplier * self.atr)
        self.lower_band = self.I(lambda: self.midline - self.volatility_multiplier * self.atr)
        
        # 🕯️ CANDLE PATTERN DETECTION 🕯️
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        
        # 🌙 LONG ENTRY CONDITION 🌙
        if (price <= lower * 1.005 and 
            self.hammer[-1] == 100 and 
            self.swing_lows[-1] > self.swing_lows[-2]):
            
            sl = self.swing_lows[-1] - 0.5 * self.atr[-1]
            risk_amount = self.equity * self.risk_pct
            risk_per_share = price - sl
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl, tp=price + 3*risk_per_share)
                print(f"🌙 MOON DEV LONG SIGNAL 🌙 | Entry: {price:.2f} | SL: {sl:.2f} | TP: {price + 3*risk_per_share:.2f} | Size: {position_size} 🚀")

        # 🌑 SHORT ENTRY CONDITION 🌑
        elif (price >= upper * 0.995 and 
              self.shooting_star[-1] == -100 and 
              self.swing_highs[-1] < self.swing_highs[-2]):
            
            sl = self.swing_highs[-1] + 0.5 * self.atr[-1]
            risk_amount = self.equity * self.risk_pct
            risk_per_share = sl - price
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.sell(size=position_size, sl=sl, tp=price - 3*risk_per_share)
                print(f"🌑 MOON DEV SHORT SIGNAL 🌑 | Entry: {price:.2f} | SL: {sl:.2f} | TP: {price - 3*risk_per_share:.2f} | Size: {position_size} 🌑")

# 🌕 RUN