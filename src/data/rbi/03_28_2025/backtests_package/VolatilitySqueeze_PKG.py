Here's the fixed code with all backtesting.lib references removed and Moon Dev themed enhancements:

```python
# 🌙 Moon Dev's Volatility Squeeze Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VolatilitySqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Convert day periods to 15m intervals (1 day = 96 periods)
        self.atr_period_10 = 10 * 96
        self.atr_period_50 = 50 * 96
        self.atr_period_20 = 20 * 96
        self.consolidation_period = 5 * 96
        
        # Calculate indicators using TA-Lib with self.I()
        self.atr10 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period_10)
        self.atr50_sma = self.I(talib.SMA, self.atr10, timeperiod=self.atr_period_50)
        self.atr20_sma = self.I(talib.SMA, self.atr10, timeperiod=self.atr_period_20)
        self.consolidation_high = self.I(talib.MAX, self.data.High, timeperiod=self.consolidation_period)
        self.consolidation_low = self.I(talib.MIN, self.data.Low, timeperiod=self.consolidation_period)
        
        print("🌙✨ MOON DEV: Indicators initialized with cosmic precision! No backtesting.lib used - pure lunar energy!")

    def next(self):
        current_close = self.data.Close[-1]
        
        # 🌙 Volatility Contraction Check
        volatility_contraction = (self.atr10[-1] < self.atr50_sma[-1])
        
        if not self.position:
            # 🚀 Long Entry Logic
            if (volatility_contraction and 
                current_close > self.consolidation_high[-1]):
                
                stop_price = self.consolidation_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🌙🚀 MOON DEV LONG ENTRY at {current_close:.2f}!"
                          f" Size: {position_size} | Cosmic Stop: {stop_price:.2f}")
            
            # 🪐 Short Entry Logic
            elif (volatility_contraction and 
                  current_close < self.consolidation_low[-1]):
                
                stop_price = self.consolidation_high[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = stop_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.sell(size=position_size, sl=stop_price)
                    print(f"🌙🌑 MOON DEV SHORT ENTRY at {current_close:.2f}!"
                          f" Size: {position_size} | Lunar Stop: {stop_price:.2f}")
        
        # ✨ Exit Logic - Volatility Expansion
        elif self.atr10[-1] > self.atr20_sma[-1]:
            self.position.close()
            print(f"🌙💫 MOON DEV EXIT: Volatility expansion detected at {current_close:.2f}!")

# 🌌 Data Preparation Ritual
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

print("🌙📊 MOON DEV: Data cleansed with lunar energy! First 3 samples:")
print(data.head(3))

# 🚀 Launch Backtest
bt = Backtest(data, VolatilitySqueeze, cash=1_000_000, exclusive_orders=True)