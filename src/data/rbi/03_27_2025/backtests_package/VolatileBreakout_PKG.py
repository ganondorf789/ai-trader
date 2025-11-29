Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolatileBreakout Backtest Script 🚀✨

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatileBreakout(Strategy):
    ema_period_short = 50
    ema_period_long = 200
    atr_period = 14
    atr_ma_period = 20
    volume_ma_period = 20
    recent_swing_period = 5
    risk_pct = 0.01  # 1% risk per trade 🌕

    def init(self):
        # Trend indicators 🌗
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema_period_short)
        self.ema200 = self.I(talib.EMA, self.data.Close, self.ema_period_long)
        
        # Volatility indicators 🌊
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_ma_period)
        
        # Volume indicators 📊
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # Swing levels 🔼🔽
        self.swing_high = self.I(talib.MAX, self.data.High, self.recent_swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.recent_swing_period)

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Debug Prints 🌙✨
        print(f"\n🌕 Candle: {self.data.index[-1]} | Price: {current_close:.2f}")
        print(f"✨ EMA50: {self.ema50[-1]:.2f} vs EMA200: {self.ema200[-1]:.2f}")
        print(f"🌊 ATR: {self.atr[-1]:.2f} vs ATR MA: {self.atr_ma[-1]:.2f}")
        print(f"📈 Volume: {current_volume:.2f} vs Vol MA: {self.volume_ma[-1]:.2f}")

        if not self.position:
            # Long Entry Logic 🟢
            if (self.ema50[-1] > self.ema200[-1] and
                self.atr[-1] < self.atr_ma[-1] and
                current_volume > 2 * self.volume_ma[-1] and
                current_close > self.swing_high[-1]):
                
                sl = current_close - self.atr[-1]
                tp = current_close + 2 * self.atr[-1]
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (current_close - sl)))
                
                print(f"\n🚀🌕 BULLISH BREAKOUT! 🚀✨")
                print(f"Entry: {current_close:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                print(f"Risk: {risk_amount:.2f} | Size: {position_size} units")
                
                self.buy(size=position_size, sl=sl, tp=tp)

            # Short Entry Logic 🔴
            elif (self.ema50[-1] < self.ema200[-1] and
                  self.atr[-1] < self.atr_ma[-1] and
                  current_volume > 2 * self.volume_ma[-1] and
                  current_close < self.swing_low[-1]):
                
                sl = current_close + self.atr[-1]
                tp = current_close - 2 * self.atr[-1]
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (sl - current_close)))
                
                print(f"\n🚀🌕 BEARISH BREAKDOWN! 🚀✨")
                print(f"Entry: {current_close:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                print(f"Risk: {risk_amount:.2f} | Size: {position_size} units")
                
                self.sell(size=position_size, sl=sl, tp=tp)
        else: