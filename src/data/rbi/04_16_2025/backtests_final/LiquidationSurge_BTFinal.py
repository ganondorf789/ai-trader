I'll fix the incomplete code while maintaining the original strategy logic and adding Moon Dev themed debug prints. Here's the complete fixed version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquidationSurge(Strategy):
    # Lunar configuration 🌙
    swing_period = 20
    atr_period = 14
    atr_ma_period = 20
    volume_multiplier = 1.5
    risk_pct = 0.01
    rr_ratio = 2
    volatility_threshold = 0.995  # 0.5% approach threshold

    def init(self):
        # Celestial indicator calculations 🌌
        print("🌌 Initializing Moonbase Indicators...")
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING LOW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_ma_period, name='ATR_MA')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='VOLUME_MA')
        print("✨ Moonbase Indicators Online! Ready for Lunar Trading! 🌙")

    def next(self):
        # Moonbase security check 🛸
        if len(self.data.Close) < 50:  # Ensure enough data
            print("🛡️ Moonbase Security Alert: Insufficient Data for Analysis")
            return

        # Cosmic metrics 🌠
        price = self.data.Close[-1]
        swing_h = self.swing_high[-1]
        swing_l = self.swing_low[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.volume_ma[-1]

        # Lunar position sizing calculator 🌝
        def moon_size(entry, sl):
            risk_amount = self.risk_pct * self.equity
            risk_per_share = abs(entry - sl)
            return int(round(risk_amount / risk_per_share)) if risk_per_share > 0 else 0

        # Main trading logic 🚀
        if not self.position:
            # Long entry constellation 🌟
            if (self.data.High[-1] >= swing_h * self.volatility_threshold and 
                atr < atr_ma and 
                vol > vol_ma * self.volume_multiplier):
                
                # Replaced crossover with manual check
                if self.data.Close[-2] < swing_h and self.data.Close[-1] > swing_h:
                    sl = self.data.Low[-1]
                    size = moon_size(price, sl)
                    if size > 0:
                        tp = price + (price - sl) * self.rr_ratio
                        print(f"🌙✨ LONG SIGNAL :: Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} :: Moon Rocket Ignition! 🚀")
                        self.buy(size=size, sl=sl, tp=tp)

            # Short entry constellation 🌑
            elif (self.data.Low[-1] <= swing_l * (2 - self.volatility_threshold) and 
                  atr < atr_ma and 
                  vol > vol_ma * self.volume_multiplier):
                
                # Replaced crossunder with manual check
                if self.data.Close[-2] > swing_l and self.data.Close[-1] < swing_l:
                    sl = self.data.High[-1]
                    size = moon_size(price, sl)
                    if size > 0:
                        tp = price - (sl - price) * self.rr_ratio
                        print(f"🌙🌑 SHORT SIGNAL :: Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} :: Dark Moon Gravitational Pull! 🌌")
                        self.sell(size=size, sl=sl, tp=tp)
        else:
            # Moon landing profit taking 🌕
            if self.position.is_long and self.data.Close[-1] >= self.position.tp:
                print(f"🎯🌕 LONG TP HIT! Cosmic profits secured at {self.data