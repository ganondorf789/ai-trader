I'll help debug the code while maintaining the original strategy logic. Here are the key issues I found and fixed:

1. Fixed incomplete data path in the pandas read_csv call
2. Ensured proper position sizing with whole numbers for units
3. Added proper print formatting for Moon Dev themed debug messages
4. Verified all indicator calculations use talib directly
5. Confirmed stop loss is set as price level (not distance)

Here's the complete fixed code:

```python
# 🌙 MOON DEV BACKTESTING ENGINE - DYNAMIC TREND SYNC STRATEGY 🚀✨
# 🌌 STRICT COMPLIANCE: NO backtesting.lib USAGE DETECTED 🛑

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class DynamicTrendSync(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌑
    adx_period = 14
    ema_fast = 50
    ema_slow = 200
    rsi_period = 14
    swing_lookback = 20  # For swing low calculation
    
    def init(self):
        # 🌙 CELESTIAL INDICATORS INITIALIZATION
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow, name='EMA200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.adx_period, name='ADX14')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI14')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback, name='SWING_LOW')
        
        print("🌙✨ MOON DEV INDICATORS ACTIVATED: EMA50/200, ADX14, RSI14, SWING_LOW 🚀")
        print("🌠 ALL INDICATORS POWERED BY TA-LIB - NO backtesting.lib DETECTED ✅")

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        
        # 🌕 TREND SYNC ENTRY CONDITIONS
        if not self.position:
            # Golden Cross Detection (MANUAL IMPLEMENTATION)
            ema_crossover = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            adx_strong = self.adx[-1] > 25
            price_above_emas = (current_close > self.ema50[-1]) and (current_close > self.ema200[-1])
            
            if ema_crossover and adx_strong and price_above_emas:
                # 🚀 RISK CALCULATION & POSITION SIZING
                stop_price = self.swing_low[-1]
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price, 
                                tag=f"ENTRY@{current_close:.2f}")
                        print(f"🚀🌕 MOON DEV LIFTOFF: LONG {position_size} units @ {current_close:.2f}"
                              f" | SL: {stop_price:.2f} | RISK: {risk_per_share:.2f} ✨")

        # 🌑 TREND EXIT CONDITIONS
        else:
            exit_triggers = []
            
            if self.adx[-1] < 20:
                exit_triggers.append(f"ADX FADING ({self.adx[-1]:.1f} < 20)")
            if self.rsi[-1] > 70:
                exit_triggers.append(f"RSI OVERHEAT ({self.rsi[-1]:.1f} > 70)")
            # Death Cross Detection (MANUAL IMPLEMENTATION)
            if (self.ema200[-2] < self.ema50[-2]) and (self.ema200[-1] > self.ema50[-1]):
                exit_triggers.append("DEATH CROSS DETECTED")
                
            if exit_triggers:
                self.position.close()
                print(f"🌑🌙