I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed prints:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV DATA PREPARATION ✨
print("🌙✨ INITIALIZING MOON DEV DATA SYSTEMS...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data according to Moon Dev specs
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
print("🌙✨ DATA SYSTEMS ONLINE! 🚀")

class LiquidationRebound(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 ICHIMOKU CLOUD CALCULATIONS ✨
        print("🌙✨ INITIALIZING MOON DEV INDICATORS...")
        self.tenkan_high = self.I(talib.MAX, self.data.High, timeperiod=9)
        self.tenkan_low = self.I(talib.MIN, self.data.Low, timeperiod=9)
        self.kijun_high = self.I(talib.MAX, self.data.High, timeperiod=26)
        self.kijun_low = self.I(talib.MIN, self.data.Low, timeperiod=26)
        self.spanB_high = self.I(talib.MAX, self.data.High, timeperiod=52)
        self.spanB_low = self.I(talib.MIN, self.data.Low, timeperiod=52)
        print("🌙✨ INDICATORS READY FOR LIFTOFF! 🚀")
        
    def next(self):
        # 🌙✨ MOON DEV INDICATOR CALCULATIONS 🚀
        if len(self.data) < 52:
            return
            
        # Ichimoku components shifted 26 periods
        tenkan = (self.tenkan_high[-26] + self.tenkan_low[-26])/2
        kijun = (self.kijun_high[-26] + self.kijun_low[-26])/2
        spanA = (tenkan + kijun)/2
        spanB = (self.spanB_high[-26] + self.spanB_low[-26])/2
        
        # 🌋 Liquidation threshold calculation
        liquidation_threshold = np.percentile(self.data.High[-96:], 90) if len(self.data) > 96 else 0
        current_liquidations = self.data.High[-1]
        
        # 📉 Orderbook imbalance zone (last hour lows)
        stop_level = np.min(self.data.Low[-4:]) if len(self.data) > 4 else 0
        
        # 🌙✨ ENTRY LOGIC 🚀
        if (not self.position and
            current_liquidations > liquidation_threshold and
            self.data.Close[-1] > max(spanA, spanB) and
            (abs(spanA - spanB) > self.data.Close[-1]*0.005)):
            
            # 🎯 RISK MANAGEMENT CALCULATIONS 🌙
            risk_amount = self.equity * self.risk_per_trade
            price_distance = self.data.Close[-1] - stop_level
            position_size = int(round(risk_amount / price_distance))
            
            if position_size > 0:
                print(f"🌙✨🚀 MOON DEV LONG SIGNAL | Price: {self.data.Close[-1]:.2f} | Size: {position_size}")
                self.buy(size=position_size, sl=stop_level, tag="LiquidationRebound")
        
        # 💰 EXIT LOGIC ✨
        if self.position:
            take_profit = max(spanA, spanB)
            if self.data.Close[-1] >= take_profit:
                print(f"🌙✨✅ PROFIT TAKEN | Price: {self.data.Close[-1]:.2f}")
                self.position.close()
            elif self.data.Low[-1] < self.position