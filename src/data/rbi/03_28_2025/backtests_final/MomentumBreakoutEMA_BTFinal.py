I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class MomentumBreakoutEMA(Strategy):
    ema50_period = 50
    ema200_period = 200
    adx_period = 14
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        self.signal_long = False
        self.signal_short = False
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Process pending signals from previous bar
        if self.signal_long and not self.position:
            risk_amount = self.equity * self.risk_pct
            atr_value = self.atr[-1]
            entry_price = self.data.Open[-1]
            
            if atr_value == 0:
                self.signal_long = False
                print("🌙⚠️ ATR ZERO ALERT! Aborting long entry")
                return
                
            position_size = risk_amount / (atr_value * 1)
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.buy(size=position_size)
                self.trailing_high = self.data.High[-1]
                print(f"🚀🌙 MOONSHOT LONG! Size: {position_size} @ {entry_price:.2f} | ATR: {atr_value:.2f}")
            self.signal_long = False

        elif self.signal_short and not self.position:
            risk_amount = self.equity * self.risk_pct
            atr_value = self.atr[-1]
            entry_price = self.data.Open[-1]
            
            if atr_value == 0:
                self.signal_short = False
                print("🌙⚠️ ATR ZERO ALERT! Aborting short entry")
                return
                
            position_size = risk_amount / (atr_value * 1)
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.sell(size=position_size)
                self.trailing_low = self.data.Low[-1]
                print(f"🌑🌙 DARK SIDE SHORT! Size: {position_size} @ {entry_price:.2f} | ATR: {atr_value:.2f}")
            self.signal_short = False

        # Detect new signals
        if (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1]) and self.adx[-1] > 25:
            print(f"✨🌙 GOLDEN CROSS DETECTED! EMA50({self.ema50[-1]:.2f}) > EMA200({self.ema200[-1]:.2f}) | ADX Strength: {self.adx[-1]:.2f}")
            self.signal_long = True
            
        elif (self.ema50[-2] > self.ema200[-2] and self.ema50[-1] < self.ema200[-1]) and self.adx[-1] > 25:
            print(f"💀🌙 DEATH CROSS DETECTED! EMA50({self.ema50[-1]:.2f}) < EMA200({self.ema200[-1]:.2f}) | ADX Strength: {self.adx[-1]:.2f}")
            self.signal_short = True

        # Manage exits
        if self.position.is_long:
            self.trailing_high = max(self.trailing_high, self.data.High[-1])
            stop_level = self.trailing_high - 2 * self.atr[-1]
            
            if self.data.Low[-1] <= stop_level:
                self.position.close()
                print(f"🌙🔻 LUNAR ECLIPSE! LONG EXIT @ {stop_level:.2f