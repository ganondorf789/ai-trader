```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquiVoltSqueeze(Strategy):
    risk_percent = 0.01
    atr_multiplier = 3
    bb_period = 20
    atr_period = 14
    swing_period = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(
            lambda: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0),
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Bollinger Band Width
        bbw = (self.upper - self.lower) / self.middle
        self.bbw = self.I(lambda: bbw, name='BBW')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing High/Low
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_period, name='Swing_Low')

        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')

        # BBW Percentiles
        self.bbw_low_threshold = self.I(
            lambda: self.bbw.rolling(252).quantile(0.05), name='BBW_5th_Percentile'
        )
        self.bbw_high_threshold = self.I(
            lambda: self.bbw.rolling(252).quantile(0.20), name='BBW_20th_Percentile'
        )

        self.trailing_stop = None
        self.highest_close = None

    def next(self):
        if not self.position:
            # Long entry conditions
            bbw_cond = self.bbw[-1] <= self.bbw_low_threshold[-1]
            in_swing_zone = self.data.Low[-1] <= self.swing_low[-1] * 1.01
            volume_cond = self.data.Volume[-1] > self.volume_ma[-1]
            touch_bb = self.data.Low[-1] <= self.lower[-1]

            if bbw_cond and in_swing_zone and volume_cond and touch_bb:
                risk_amount = self.equity * self.risk_percent
                sl = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=sl)
                        print(f"🌙 MOON DEV LONG ENTRY ✨ | Size: {size} | Entry: {self.data.Close[-1]} | SL: {sl}")

            # Short entry conditions
            bbw_cond_short = self.bbw[-1] <= self.bbw_low_threshold[-1]
            in_swing_zone_short = self.data.High[-1] >= self.swing_high[-1] * 0.99
            volume_cond_short = self.data.Volume[-1] > self.volume_ma[-1]
            touch_bb_short = self.data.High[-1] >= self.upper[-1]

            if bbw_cond_short and in_swing_zone_short and volume_cond_short and touch_bb_short:
                risk_amount = self.equity * self.risk_percent
                sl = self.swing_high[-1]
                risk_per_share = sl - self.data.Close[-1]
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.sell(size=size, sl=sl)
                        print(f"🌙 MOON DEV SHORT ENTRY ✨ | Size: {size} | Entry: {self.data.Close[-1]} | SL: {sl}")

        else:
            if self.position.is_long:
                # Trailing stop logic for long positions
                if self.data.Close[-1] > (