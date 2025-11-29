```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta
from pandas_ta import median as ta_median

class VolCompressBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    consecutive_losses = 0
    in_compression = False
    compression_high = None
    compression_low = None

    def init(self):
        # Calculate Bollinger Bands
        close = self.data.Close
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda: upper, name='Upper BB')
        self.middle_band = self.I(lambda: middle, name='Middle BB')
        self.lower_band = self.I(lambda: lower, name='Lower BB')

        # Calculate Bollinger Bandwidth
        bbw = (upper - lower) / middle
        self.bbw = self.I(bbw, name='BBW')
        self.bbw_avg = self.I(talib.SMA, bbw, 10, name='BBW Avg')

        # Calculate volume median
        self.volume_median = self.I(ta_median, self.data.Volume, 30, name='Volume Median')

    def next(self):
        # Update compression range
        if self.bbw[-1] < self.bbw_avg[-1]:
            if not self.in_compression:
                self.in_compression = True
                self.compression_high = self.data.High[-1]
                self.compression_low = self.data.Low[-1]
                print(f"🌙 MOON DEV: Compression Phase START! 📉 Range: {self.compression_high - self.compression_low:.2f}")
            else:
                self.compression_high = max(self.compression_high, self.data.High[-1])
                self.compression_low = min(self.compression_low, self.data.Low[-1])
        else:
            if self.in_compression:
                print(f"🌙 MOON DEV: Compression Phase END! 🚀")
            self.in_compression = False

        # Entry logic
        if not self.position and self.in_compression and self.consecutive_losses < 3:
            volume_ok = self.data.Volume[-1] > 2 * self.volume_median[-1]
            close_above_upper = self.data.Close[-1] > self.upper_band[-1]
            close_below_lower = self.data.Close[-1] < self.lower_band[-1]

            if volume_ok and (close_above_upper or close_below_lower):
                risk_amount = self.equity() * self.risk_per_trade
                compression_range = self.compression_high - self.compression_low

                if close_above_upper:
                    direction = 'LONG'
                    est_entry = self.data.Close[-1]
                    stop_loss = self.compression_low
                    risk_per_share = est_entry - stop_loss
                    take_profit = est_entry + 1.5 * compression_range
                else:
                    direction = 'SHORT'
                    est_entry = self.data.Close[-1]
                    stop_loss = self.compression_high
                    risk_per_share = stop_loss - est_entry
                    take_profit = est_entry - 1.5 * compression_range

                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        if direction == 'LONG':
                            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                        else:
                            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                        print(f"🌙 MOON DEV {direction} ENTRY! 🚀 Size: {position_size}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}")

        # Volume-based exit
        if self.position and self.data.Volume[-1] < self.volume_median[-1]:
            self.position.close()
            print(f"🌙 MOON DEV VOLUME EXIT! 📉 Current Vol: {self.data.Volume[-1]:.2f}")

    def notify_trade(self, trade):
        if trade.is_closed:
            if trade.pnl < 0:
                self.consecutive_losses += 1
                print(f"🌙 MOON DEV LOSS! 💔 Consecutive: {self.consecutive_losses}")
            else: