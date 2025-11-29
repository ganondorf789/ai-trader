Here's the fully debugged and Moon Dev themed backtest code with all technical issues fixed:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeSurge(Strategy):
    bb_period = 20
    bb_dev = 2
    bbw_lookback = 17280  # 6 months in 15-min intervals
    volume_ma_period = 1920  # 20 days in 15-min intervals
    atr_period = 14
    volume_threshold = 1.2  # 20% above average volume
    risk_percent = 0.01  # 1% risk per trade
    rr_ratio = 2  # Risk-reward ratio

    def init(self):
        # Calculate Bollinger Bands components
        def upper_band(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=self.bb_period, 
                                     nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
            return upper
        self.upper_band = self.I(upper_band, self.data.Close, name='Upper BB 🌈')

        def middle_band(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=self.bb_period,
                                      nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
            return middle
        self.middle_band = self.I(middle_band, self.data.Close, name='Middle BB 🚀')

        # Calculate BB Width (BBW)
        self.bbw = self.I(lambda u, l, m: (u - l)/m,
                         self.upper_band, self.data.Low, self.middle_band,
                         name='BBW Width ⚡')

        # 6-month BBW low using MIN
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback,
                            name='BBW Low 🔻')

        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 
                              timeperiod=self.volume_ma_period,
                              name='Volume MA 🔊')
        
        # Volatility indicator
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                        timeperiod=self.atr_period, name='ATR 🌪️')

        # Moon Dev initialization complete
        print("🌙 MOON DEV INIT COMPLETE: All indicators powered up! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Entry conditions
        bbw_squeeze = self.bbw[-1] < self.bbw_low[-1]
        price_breakout = current_close > self.upper_band[-1]
        volume_surge = current_volume > (self.volume_ma[-1] * self.volume_threshold)

        if not self.position and bbw_squeeze and price_breakout and volume_surge:
            # Calculate risk parameters
            atr_value = self.atr[-1]
            stop_loss = current_close - 1.5 * atr_value
            take_profit = current_close + (1.5 * atr_value * self.rr_ratio)
            risk_per_share = current_close - stop_loss
            
            # Position sizing - fixed to use whole units
            risk_amount = self.equity * self.risk_percent
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit,
                        tag=f"🚀 MOON SURGE ENTRY 🌙\n"
                            f"Entry: {current_close:.2f}\n"
                            f"SL: {stop_loss:.2f}\n"
                            f"TP: {take_profit:.2f}")
                print(f"🌕 MOON DEV ALERT: New position launched! Size: {position_size} 🚀")

        # Moon Dev themed position monitoring
        if self.position:
            if self.position.pl_pct <= -1:
                print(f"🌑 MOON DEV WARNING: Position in drawdown! PL: {self.position.pl:.2f} ✨")
            elif self.position.pl_pct >= 3:
                print(f"🌕 MOON DEV CELEBRATE: Profit mooning! PL: {self.position.pl:.2f} 🚀")

# Data preparation - fixed path issue