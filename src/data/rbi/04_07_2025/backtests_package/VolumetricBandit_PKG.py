Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class VolumetricBandit(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    bb_period = 20
    obv_period = 10
    obv_ma_period = 5
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        close = self.data.Close
        volume = self.data.Volume
        
        # On-Balance Volume
        self.obv = self.I(talib.OBV, close, volume, name='OBV 🌊')
        
        # OBV 10-day high
        self.obv_high = self.I(talib.MAX, self.obv, timeperiod=self.obv_period, name='OBV High 🔺')
        
        # OBV 5-day SMA
        self.obv_ma = self.I(talib.SMA, self.obv, timeperiod=self.obv_ma_period, name='OBV MA 📉')
        
        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(talib.BBANDS, close, 
                                                          timeperiod=self.bb_period,
                                                          nbdevup=2, nbdevdn=2,
                                                          name=['BB Upper 🌈', 'BB Mid 🎯', 'BB Lower 🌊'])
        
        # Swing lows (20-period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low 🔻')
        
    def next(self):
        price = self.data.Close[-1]
        obv = self.obv[-1]
        obv_high = self.obv_high[-1]
        obv_ma = self.obv_ma[-1]
        
        # Calculate price position in Bollinger Bands
        bb_width = self.bb_upper[-1] - self.bb_lower[-1]
        if bb_width == 0:  # Avoid division by zero
            return
            
        price_position = (price - self.bb_lower[-1]) / bb_width
        
        # Entry conditions
        if not self.position:
            if (obv >= obv_high and 
                price_position <= 0.2 and 
                bb_width > np.percentile(bb_width, 30)):
                
                # Calculate stop loss
                stop_loss = min(self.swing_low[-1], self.bb_lower[-1])
                risk_per_share = price - stop_loss
                
                if risk_per_share <= 0:
                    return
                
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / risk_per_share))
                
                # Moon-themed debug print
                print(f"🌕 LUNAR ENTRY SIGNAL 🌙")
                print(f"✨ Price: {price:.2f} | OBV: {obv:.2f} (10D High: {obv_high:.2f})")
                print(f"🎯 BB Position: {price_position*100:.1f}% | Size: {position_size} shares")
                
                self.buy(size=position_size, sl=stop_loss, 
                        tag="Volumetric Bandit Entry 🚀")
        
        # Exit conditions
        else:
            # Replaced crossover with direct comparison
            if price[-2] < self.bb_upper[-2] and price[-1] > self.bb_upper[-1]:
                print(f"🌑 EXIT SIGNAL: Price touched Upper BB 🌈")
                self.position.close()
            elif obv_ma[-2] > self.obv[-2] and obv_ma[-1] < self.obv[-1]:
                print(f"🌑 EXIT SIGNAL: OBV crossed below MA 📉")
                self.position.close()

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index