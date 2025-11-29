I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VortexDivergence(Strategy):
    vi_period = 14
    cmf_period = 20
    volume_ma_period = 20
    risk_per_trade = 0.02
    swing_period = 20

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])

        # Calculate indicators
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(
            lambda: talib.VORTEX(high, low, close, self.vi_period),
            name=['VI_Plus', 'VI_Minus']
        )

        # Chaikin Money Flow
        self.cmf = self.I(
            lambda: ta.cmf(high, low, close, volume, length=self.cmf_period),
            name='CMF'
        )

        # Volume MA
        self.volume_ma = self.I(
            talib.SMA, volume, timeperiod=self.volume_ma_period,
            name='Volume_MA'
        )

        # Swing High/Low
        self.swing_high = self.I(
            talib.MAX, high, timeperiod=self.swing_period,
            name='Swing_High'
        )
        self.swing_low = self.I(
            talib.MIN, low, timeperiod=self.swing_period,
            name='Swing_Low'
        )

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        if not self.position:
            # Long Entry Conditions
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
                self.data.Low[-1] < self.data.Low[-2] and
                self.cmf[-1] > self.cmf[-2] and
                current_volume > self.volume_ma[-1]):

                sl_price = self.swing_low[-1]
                risk_per_share = current_close - sl_price
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        print(f"🌙✨🚀 MOON SHOT! LONG ENTRY at {current_close:.2f} | Size: {position_size} | SL: {sl_price:.2f}")

            # Short Entry Conditions
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and
                  self.data.High[-1] > self.data.High[-2] and
                  self.cmf[-1] < self.cmf[-2] and
                  current_volume > self.volume_ma[-1]):

                sl_price = self.swing_high[-1]
                risk_per_share = sl_price - current_close
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        print(f"🌙✨💥 SHORT CIRCUIT! SHORT ENTRY at {current_close:.2f} | Size: {position_size} | SL: {sl_price:.2f}")

        else:
            # Exit Conditions
            if self.position.is_long:
                if ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) or
                    current_volume > 2 * self.volume_ma[-1]):
                    self.position.close()
                    print(f"🌙💫📉 LUNAR ECLIPSE! EXIT LONG at {current_close:.2f}")

            elif self.position