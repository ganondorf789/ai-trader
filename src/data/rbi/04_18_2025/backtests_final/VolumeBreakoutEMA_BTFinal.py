I'll analyze and debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLUMEBREAKOUTEMA STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumeBreakoutEMA(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    
    def init(self):
        # 🌟 INDICATORS USING TA-LIB WITH self.I() WRAPPER
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.volume_ma5 = self.I(talib.SMA, self.data.Volume, timeperiod=5)
        self.consolidation_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        
        # ✨ TRADE TRACKING VARIABLES
        self.entry_price = None
        self.initial_stop = None
        self.trailing_high = None

    def next(self):
        # 🌌 MOON DEV DEBUG: Print current equity status
        if len(self.data) % 1000 == 0:
            print(f"🌙 Current Equity: {self.equity:,.2f} | Bar: {len(self.data)}")
        
        if not self.position:
            # 🚀 ENTRY CONDITIONS
            golden_cross = (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1])
            volume_decline = (self.data.Volume[-1] < self.data.Volume[-2] < 
                             self.data.Volume[-3] < self.data.Volume[-4])
            close_above_high = self.data.Close[-1] > self.consolidation_high[-1]
            
            if golden_cross and volume_decline and close_above_high:
                # 🌕 CALCULATE POSITION SIZE
                atr_value = self.atr14[-1]
                self.entry_price = self.data.Close[-1]
                self.initial_stop = self.entry_price - 2 * atr_value
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.entry_price - self.initial_stop
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"\n🌙✨ MOON DEV ENTRY ✨🌙\n"
                          f"Entry Price: {self.entry_price:.2f}\n"
                          f"Position Size: {position_size} units\n"
                          f"Initial Stop: {self.initial_stop:.2f}\n"
                          f"Equity: {self.equity:,.2f}\n{'='*30}")
                    
                    # Initialize trailing high
                    self.trailing_high = self.entry_price
        else:
            # 🌑 UPDATE TRAILING HIGH
            self.trailing_high = max(self.trailing_high, self.data.High[-1])
            
            # 🛑 CHECK EXIT CONDITIONS
            current_close = self.data.Close[-1]
            
            # Trailing stop logic
            if current_close >= self.entry_price * 1.05:
                trailing_stop = self.trailing_high * 0.95
                if current_close < trailing_stop:
                    self.position.close()
                    print(f"\n🚀💫 MOON DEV TRAILING EXIT 💫🚀\n"
                          f"Exit Price: {current_close:.2f}\n"
                          f"Profit: {self.position.pl_pct:.2%}\n"
                          f"Equity: {self.equity:,.2f}\n{'='*30}")
            
            # Initial stop loss check
            elif current_close < self.initial_stop:
                self.position.close()
                print(f"\n🌙🛑 MOON DEV STOP LOSS EXIT 🛑🌙\n"
                      f"Exit Price: {current_close:.2f}\n"
                      f"Loss: {self.position.pl_pct:.2%}\n"
                      f"Equity: {self.equity:,.2f}\n{'='*30}")

#