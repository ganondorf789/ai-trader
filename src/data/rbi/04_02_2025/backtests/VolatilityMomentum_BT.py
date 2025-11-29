Here's the complete implementation of the VolatilityMomentum strategy for backtesting.py:

```python
# 🌙 Moon Dev's Volatility Momentum Backtest 🌙
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatilityMomentum(Strategy):
    cmo_period = 14
    volatility_period = 20
    adjustment_factor = 1.5
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌟 Core Indicators 🌟
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=self.cmo_period)
        
        # 💹 Volatility Calculation
        returns = self.I(lambda close: close.pct_change() * 100, self.data.Close)
        self.volatility = self.I(talib.STDDEV, returns, timeperiod=self.volatility_period)
        
        # 📈 Bollinger Bands with %B
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                    timeperiod=self.bb_period, nbdevup=self.bb_dev,
                                                    nbdevdn=self.bb_dev, matype=0)
        self.b_pct = self.I(lambda c, u, l: (c - l) / (u - l + 1e-10),  # Avoid division by zero
                          self.data.Close, self.upper, self.lower)
        
        # 🛑 Stop Loss Tools
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # 🌙 Debugging Series
        self.threshold = self.I(lambda vol, adj: vol * adj,
                               self.volatility, np.full_like(self.volatility, self.adjustment_factor))

    def next(self):
        current_close = self.data.Close[-1]
        current_cmo = self.cmo[-1]
        current_vol = self.volatility[-1]
        
        # 🚀 Entry Logic
        if not self.position:
            # CMO crosses above dynamic threshold
            cmo_cross = crossover(self.cmo, self.threshold)
            
            # Volatility expansion check
            vol_expanding = current_vol > self.volatility[-2] if len(self.volatility) > 1 else False
            
            if cmo_cross and vol_expanding:
                # 🌙 Risk Management Calculations
                equity = self.equity
                risk_amount = equity * self.risk_pct
                atr_value = self.atr[-1]
                stop_price = max(current_close - 2*atr_value, self.swing_low[-1])
                
                risk_per_share = current_close - stop_price
                if risk_per_share <= 0:
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                position_size = min(position_size, int(equity // current_close))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🚀 MOON LAUNCH! LONG {position_size} @ {current_close:.2f} "
                          f"| SL: {stop_price:.2f} 🌙 (Risk: {risk_per_share:.2f}/share)")
        
        # 🌑 Exit Logic
        else:
            current_b = self.b_pct[-1]
            
            # Mean reversion exit
            if current_b < 0.5 and self.b_pct[-2] >= 0.5:
                self.position.close()
                print(f"🌑 EARTH'S GRAVITY! Closing @ {current_close:.2f} | %B: {current_b:.2f} ✨")
            
            # Profit taking exit
            elif current_b > 0.8:
                self.position.close()
                print(f"🌕 FULL MOON PROFITS! Closing @ {current_close:.2f} | %B: {current_b:.2f} 💰")

# 🌍 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Cleanse Data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if '