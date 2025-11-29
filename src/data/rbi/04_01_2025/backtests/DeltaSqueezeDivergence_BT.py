```python
# 🌙 Moon Dev's DeltaSqueezeDivergence Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

class DeltaSqueezeDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 20
    swing_period = 20
    divergence_lookback = 3

    def init(self):
        # 🌙 Calculate core indicators
        self.ema = self.I(talib.EMA, self.data.Close, self.atr_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.avg_atr = self.I(talib.SMA, self.atr, self.atr_period)
        
        # 🌟 Calculate delta-based indicators
        self.cum_delta = self.I(self.calculate_cumulative_delta)
        self.price_swing_high = self.I(talib.MAX, self.data.High, self.divergence_lookback)
        self.delta_swing_high = self.I(talib.MAX, self.cum_delta, self.divergence_lookback)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        
        # ✨ Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)

    def calculate_cumulative_delta(self):
        # 🌙 Calculate delta from price/volume relationship
        delta = np.where(self.data.Close > self.data.Open, self.data.Volume,
                        np.where(self.data.Close < self.data.Open, -self.data.Volume, 0))
        return delta.cumsum()

    def next(self):
        current_close = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]

        # 🌙 Keltner Channel calculations
        lower_kc = self.ema[-1] - 1.5 * self.atr[-1]
        upper_kc = self.ema[-1] + 1.5 * self.atr[-1]
        kc_width = upper_kc - lower_kc

        if not self.position:
            # 🚀 ENTRY CONDITIONS
            kc_contraction = self.atr[-1] <= 0.5 * self.avg_atr[-1]
            
            # 🌟 Divergence check
            price_divergence = (self.price_swing_high[-1] > self.price_swing_high[-2] and
                               self.delta_swing_high[-1] < self.delta_swing_high[-2])
            
            # ✨ Price breakdown
            price_breakout = current_close < lower_kc

            if kc_contraction and price_divergence and price_breakout:
                # 🌙 RISK MANAGEMENT
                stop_loss = self.swing_high[-1]
                risk_amount = self.equity * self.risk_percent
                risk_per_share = stop_loss - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    position_size = min(position_size, int(self.equity // current_close))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_loss, tag="Short Entry")
                        print(f"🌙🚀 MOON DEV SHORT ENTRY | Size: {position_size} | Entry: {current_close} | SL: {stop_loss}")

        else:
            # 💫 EXIT CONDITIONS
            if self.position.is_short:
                # ✨ Primary exit: Price closes back inside KC
                if lower_kc < current_close < upper_kc:
                    self.position.close()
                    print(f"🌙✨ PRIMARY EXIT | Price: {current_close}")

                # 🌟 Secondary exit: Positive delta divergence
                delta_swing_low = self.I(talib.MIN, self.cum_delta, 3)
                price_swing_low = self.I(talib.MIN, self.data.Low, 3)
                if (price_swing_low[-1] < price_swing_low[-2] and
                    delta_swing_low[-1] > delta_swing_low[-2]):
                    self.position.close()
                    print(f"🌙💫 DIVERGENCE EXIT | Price: {current_close}")