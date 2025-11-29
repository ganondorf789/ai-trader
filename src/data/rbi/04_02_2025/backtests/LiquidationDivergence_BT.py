```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and clean data
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

class LiquidationDivergence(Strategy):
    risk_pct = 0.02
    rsi_period = 14
    swing_window = 20
    oi_lookback = 8
    divergence_period = 5
    
    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.oi_max = self.I(talib.MAX, self.data.OpenInterest, self.oi_lookback)
        
    def next(self):
        if len(self.data.Close) < 50:
            return
            
        current_oi = self.data.OpenInterest[-1]
        oi_drop = (self.oi_max[-1] - current_oi)/self.oi_max[-1] if self.oi_max[-1] != 0 else 0
        
        # Moon Dev Divergence Detection 🌙
        price_lows = self.data.Low[-self.divergence_period:]
        rsi_lows = self.rsi[-self.divergence_period:]
        price_highs = self.data.High[-self.divergence_period:]
        rsi_highs = self.rsi[-self.divergence_period:]
        
        bull_div = all(price_lows[i] > price_lows[i+1] and rsi_lows[i] < rsi_lows[i+1] 
                    for i in range(len(price_lows)-1))
        bear_div = all(price_highs[i] < price_highs[i+1] and rsi_highs[i] > rsi_highs[i+1] 
                    for i in range(len(price_highs)-1))
        
        # Moon Dev Liquidation Signals 🌊
        if oi_drop >= 0.2 and bull_div and not self.position:
            if self.data.Close[-1] > self.data.High[-2]:
                sl = self.swing_low[-1] * 0.99
                risk = self.data.Close[-1] - sl
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size > 0:
                    print(f"🌙 MOON DEV LONG ENTRY 🌙 | Price: {self.data.Close[-1]} | Risk: {risk:.2f}")
                    self.buy(size=position_size, sl=sl, 
                            tp=self.data.Close[-1] + 1.5*risk)
        
        elif current_oi >= self.oi_max[-1] * 1.2 and bear_div and not self.position:
            if self.data.Close[-1] < self.data.Low[-2]:
                sl = self.swing_high[-1] * 1.01
                risk = sl - self.data.Close[-1]
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size > 0:
                    print(f"🌙 MOON DEV SHORT ENTRY 🌙 | Price: {self.data.Close[-1]} | Risk: {risk:.2f}")
                    self.sell(size=position_size, sl=sl, 
                             tp=self.data.Close[-1] - 1.5*risk)
        
        # Moon Dev Exit Management 🚀
        for trade in self.trades:
            if trade.is_long and self.rsi[-1] >= 70:
                print(f"🌑 MOON DEV LONG EXIT 🌑 | RSI: {self.rsi[-1]:.2f}")
                trade.close()
            elif trade.is_short and self.rsi[-1] <= 30:
                print(f"🌑 MOON DEV SHORT EXIT 🌑 | RSI: {self.rsi[-1]:.2f}")