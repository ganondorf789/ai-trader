To address the problem of creating a backtesting implementation for the VolcanicBreakout strategy, we need to follow the specified requirements and integrate the necessary components. Here's the structured solution:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib

class VolcanicBreakout(Strategy):
    risk_percent = 0.01
    swing_period = 20
    bb_period = 20
    bb_dev = 2
    volume_sma_period = 20
    squeeze_threshold = 0.02

    def init(self):
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0),
            self.data.Close,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period, name='Volume_SMA')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing_Low')
        self.consecutive_losses = 0
        self.entry_allowed = True

    def next(self):
        if len(self.data) < self.swing_period + 1:
            return

        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]
        volume_sma = self.volume_sma[-1]

        volume_surge = volume >= 3 * volume_sma
        squeeze = (bb_upper - bb_lower) / bb_middle <= self.squeeze_threshold

        if not self.position and self.entry_allowed:
            if volume_surge and squeeze:
                if close > bb_upper:
                    entry_price = close
                    stop_loss = low
                    risk_per_unit = entry_price - stop_loss
                    if risk_per_unit <= 0:
                        return
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size == 0:
                        return
                    take_profit = entry_price + 1.5 * risk_per_unit
                    self.buy(size=position_size, sl=stop_loss, tag={'take_profit': take_profit})
                    print(f"🌋🚀 LONG entry at {entry_price}, size {position_size}, SL {stop_loss}, TP {take_profit}")
                elif close < bb_lower:
                    entry_price = close
                    stop_loss = high
                    risk_per_unit = stop_loss - entry_price
                    if risk_per_unit <= 0:
                        return
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size == 0:
                        return
                    take_profit = entry_price - 1.5 * risk_per_unit
                    self.sell(size=position_size, sl=stop_loss, tag={'take_profit': take_profit})
                    print(f"🌋🚀 SHORT entry at {entry_price}, size {position_size}, SL {stop_loss}, TP {take_profit}")

        for trade in self.trades:
            if not trade.is_closed:
                take_profit = trade.tag.get('take_profit')
                if take_profit:
                    if trade.is_long and high >= take_profit:
                        size_to_close = trade.size // 2
                        if size_to_close > 0:
                            self.position.close(size_to_close)
                            print(f"🌋🎯 Partial profit taken at {take_profit}")
                    elif trade.is_short and low <= take_profit:
                        size_to_close = trade.size // 2
                        if size_to_close > 0:
                            self.position.close(size_to_close)
                            print(f"🌋🎯 Partial profit taken at {take_profit}")

        if len(self.swing_low) >= 2 and len(self.swing_high) >= 2:
            swing_low_prev = self.swing_low[-2]
            swing_high_prev = self.swing_high[-2]
            if self.position.is_long and low