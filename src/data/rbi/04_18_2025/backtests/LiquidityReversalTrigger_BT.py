```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
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

class LiquidityReversalTrigger(Strategy):
    risk_pct = 0.01
    atr_period = 14
    swing_window = 20
    cluster_threshold = 0.005
    
    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_5day = self.I(talib.SMA, self.atr, 480)  # 5-day average for 15m data
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
    def next(self):
        moon_emoji = '🌙' * (np.random.randint(1,4)) + '✨'  # Random moon sparkles
        
        if not self.position:
            current_swing = self.swing_high[-1]
            price_approach = self.data.High[-1] >= current_swing * (1 - self.cluster_threshold)
            bearish_confirmation = self.data.Close[-1] < self.data.Close[-2]
            
            if price_approach and bearish_confirmation:
                entry_price = self.data.Close[-1]
                stop_price = current_swing * 1.01
                risk_per_share = stop_price - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.risk_pct * self.equity) / risk_per_share))
                    if position_size > 0:
                        self.entry_price = entry_price
                        self.stop_price = stop_price
                        self.tp_price = entry_price - 1.5 * risk_per_share
                        self.current_cluster = current_swing
                        
                        self.sell(size=position_size)
                        print(f"{moon_emoji} SHORT ENTERED! Size: {position_size} @ {entry_price}")
                        print(f"   🎯 TP: {self.tp_price:.2f} | 🛑 SL: {stop_price:.2f}")
        else:
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            
            # Exit conditions
            if (current_high >= self.stop_price) or \
               (current_low <= self.tp_price) or \
               (current_high > self.current_cluster * 1.02) or \
               (self.atr[-1] > self.atr_5day[-1]):
                
                print(f"{moon_emoji} EXIT TRIGGERED!")
                if current_high >= self.stop_price:
                    print(f"   🛑 STOP LOSS HIT! ({self.stop_price:.2f})")
                elif current_low <= self.tp_price:
                    print(f"   🎯 TAKE PROFIT REACHED! ({self.tp_price:.2f})")
                elif current_high > self.current_cluster * 1.02:
                    print(f"   🚨 EMERGENCY EXIT! Cluster Break: {self.current_cluster*1.02:.2f}")
                else:
                    print(f"   📉 ATR SPIKE DETECTED! ({self.atr[-1]:.2f} > {self.atr_5day[-1]:.2f})")
                
                self.position.close()

# Run backtest with $1M initial equity
bt = Backtest(data, LiquidityReversalTrigger, cash=1_000_000)
stats = bt.run()
print("\n" + "="*50)
print("🌕🌖🌗 MOON DEV BACKTEST RESULTS 🌘🌑🌒")
print("="*50)
print(stats