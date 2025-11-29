```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatileReversion(Strategy):
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 960)  # 10-day SMA for 15m data
        self.vwap = self.I(self._compute_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20)
        
        # Weekly high/low tracking
        self.weekly_high = None
        self.weekly_low = None
        self.last_week = None

    def _compute_vwap(self, high, low, close, volume, period):
        typical_price = (high + low + close) / 3
        tpv = typical_price * volume
        sum_tpv = talib.SUM(tpv, period)
        sum_vol = talib.SUM(volume, period)
        return sum_tpv / sum_vol

    def next(self):
        # Update weekly high/low
        current_dt = self.data.index[-1]
        year, week, _ = current_dt.isocalendar()
        
        if not self.last_week or (year, week) != self.last_week:
            self.weekly_high = self.data.High[-1]
            self.weekly_low = self.data.Low[-1]
            self.last_week = (year, week)
            print(f"🌙 New Week! {current_dt.date()} | High: {self.weekly_high:.2f}, Low: {self.weekly_low:.2f}")
        else:
            self.weekly_high = max(self.weekly_high, self.data.High[-1])
            self.weekly_low = min(self.weekly_low, self.data.Low[-1])

        # Skip if not enough data
        if len(self.data) < 5 or len(self.rsi) < 5:
            return

        # Get current values
        atr = self.atr[-1]
        price_close = self.data.Close[-1]

        if not self.position:
            # Long Entry Logic
            long_trigger = self.data.Low[-1] <= (self.weekly_low - 2*atr)
            bull_div = (self.data.Low[-1] < self.data.Low[-5]) and (self.rsi[-1] > self.rsi[-5])
            vol_ok = self.data.Volume[-1] < self.volume_sma[-1]
            
            if long_trigger and bull_div and vol_ok:
                risk_amount = self.equity * 0.01  # 1% risk
                entry_price = price_close
                sl = entry_price - 1.5*atr
                risk_per_unit = entry_price - sl
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=size, sl=sl, tp=self.vwap[-1])
                    print(f"🌙🚀 LONG | Entry: {entry_price:.2f} | SL: {sl:.2f} | TP: {self.vwap[-1]:.2f} | Size: {size}")

            # Short Entry Logic
            short_trigger = self.data.High[-1] >= (self.weekly_high + 2*atr)
            bear_div = (self.data.High[-1] > self.data.High[-5]) and (self.rsi[-1] < self.rsi[-5])
            vol_ok_short = self.data.Volume[-1] < self.volume_sma[-1]
            
            if short_trigger and bear_div and vol_ok_short:
                risk_amount = self.equity * 0.01  # 1% risk
                entry_price = price_close
                sl = entry_price + 1.5*atr
                risk_per_unit = sl - entry_price
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.sell(size=size, sl=sl, tp=self.vwap[-1])
                    print(f"🌙🌑 SHORT | Entry: {entry_price:.2f} | SL: {sl:.2f} | TP: {self.vwap[-1]:.2f} | Size: {size}