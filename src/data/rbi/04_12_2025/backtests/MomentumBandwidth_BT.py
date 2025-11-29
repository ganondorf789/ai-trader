```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

class MomentumBandwidth(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)

        # Indicator calculations using self.I()
        self.cmo = self.I(talib.CMO, self.data.Close, 14, name='CMO')
        self.middle_band = self.I(talib.SMA, self.data.Close, 20, name='MiddleBand')
        self.stddev = self.I(talib.STDDEV, self.data.Close, 20, name='StdDev')
        self.upper_band = self.I(lambda m, s: m + 2*s, self.middle_band, self.stddev, name='UpperBand')
        self.lower_band = self.I(lambda m, s: m - 2*s, self.middle_band, self.stddev, name='LowerBand')
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, self.upper_band, self.lower_band, self.middle_band, name='Bandwidth')
        self.bandwidth_sma5 = self.I(talib.SMA, self.bandwidth, 5, name='Bandwidth_SMA5')
        self.bandwidth_sma20 = self.I(talib.SMA, self.bandwidth, 20, name='Bandwidth_SMA20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')

        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Moon Dev debug prints
        print(f"🌙 CMO: {self.data.CMO[-1]:.1f} | BW: {self.data.Bandwidth[-1]:.4f} | SMA5: {self.data.Bandwidth_SMA5[-1]:.4f} | SMA20: {self.data.Bandwidth_SMA20[-1]:.4f} | ATR: {self.data.ATR[-1]:.2f}")

        if not self.position:
            # Long entry logic
            if crossover(self.data.CMO[-1], 50) and self.data.Bandwidth[-1] < self.data.Bandwidth_SMA5[-1]:
                risk_amount = 0.01 * self.equity
                atr = self.data.ATR[-1]
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - 2 * atr
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.trailing_high = self.data.High[-1]
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🚀🌙 LONG ENTRY! Size: {position_size} | Entry: {entry_price} | SL: {stop_loss:.2f}")

            # Short entry logic
            elif crossunder(self.data.CMO[-1], -50) and self.data.Bandwidth[-1] < self.data.Bandwidth_SMA5[-1]:
                risk_amount = 0.01 * self.equity
                atr = self.data.ATR[-1]
                entry_price = self.data.Close[-1]
                stop_loss = entry_price + 2 * atr
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.trailing_low = self.data.Low[-1]
                        self.sell(size=position_size, sl=stop_loss)
                        print(f"📉🌙 SHORT ENTRY! Size: {position_size} | Entry: {entry_price} | SL: {stop_loss:.2f}")

        else:
            # Update trailing stops
            if self.position.is_long:
                self.trailing_high = max(self.trailing_high, self.data.High[-1])
                new_sl = self.trailing_high - 2 * self.data.ATR[-1]
                if new_sl > self.position.sl:
                    self.position.sl = new_sl