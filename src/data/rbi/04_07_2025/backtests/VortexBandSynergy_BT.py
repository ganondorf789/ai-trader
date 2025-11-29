```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np

class VortexBandSynergy(Strategy):
    vi_period = 14
    bb_period = 20
    bb_std = 2
    bb_width_ma_period = 10
    atr_period = 14
    risk_pct = 0.01
    max_trade_bars = 480  # 5 days in 15m intervals

    def init(self):
        # Calculate Vortex Indicator 🌪️
        vi_plus, vi_minus = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')

        # Calculate Bollinger Bands 📊
        upper, middle, lower = talib.BBANDS(
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        self.bb_upper = self.I(lambda: upper, name='BB_UPPER')
        self.bb_middle = self.I(lambda: middle, name='BB_MID')
        self.bb_lower = self.I(lambda: lower, name='BB_LOWER')

        # Calculate Band Width Dynamics 📈
        bb_width = upper - lower
        self.bb_width = self.I(lambda: bb_width, name='BB_WIDTH')
        self.bb_width_sma = self.I(lambda: talib.SMA(bb_width, self.bb_width_ma_period), name='BB_WIDTH_SMA')

        # Calculate ATR for Risk Management 🛡️
        self.atr = self.I(lambda: talib.ATR(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
        ), name='ATR')

    def next(self):
        if self.position:
            # Moon Dev Exit Logic 🌙
            if self.position.is_long:
                if crossover(self.vi_minus, self.vi_plus) or self.bb_width[-1] < self.bb_width_sma[-1]*0.8:
                    self.position.close()
                    print(f"🌑 Moon Dev Exit LONG: Signals reversed at {self.data.Close[-1]} ✨")
            elif self.position.is_short:
                if crossover(self.vi_plus, self.vi_minus) or self.bb_width[-1] < self.bb_width_sma[-1]*0.8:
                    self.position.close()
                    print(f"🌕 Moon Dev Exit SHORT: Signals reversed at {self.data.Close[-1]} ✨")

            # Max Trade Duration Check ⌛
            if len(self.data) - self.position.entry_bar >= self.max_trade_bars:
                self.position.close()
                print(f"⏳ Moon Dev: Max trade duration reached at {self.data.Close[-1]}")

            # Trailing Stop Logic 🔄
            if self.position.is_long and self.data.High[-1] > self.position.entry_price + 1.5*self.atr[-1]:
                new_sl = self.position.entry_price
                if self.position.sl < new_sl:
                    self.position.sl = new_sl
                    print(f"🔒 Moon Dev Trailing Stop: Moved to breakeven at {new_sl}")
                    
            elif self.position.is_short and self.data.Low[-1] < self.position.entry_price - 1.5*self.atr[-1]:
                new_sl = self.position.entry_price
                if self.position.sl > new_sl:
                    self.position.sl = new_sl
                    print(f"🔒 Moon Dev Trailing Stop: Moved to breakeven at {new_sl}")
        else:
            # Moon Dev Entry Logic 🚀
            entry_price = self.data.Close[-1]
            
            # Long Conditions ✅
            long_trigger = (crossover(self.vi_plus, self.vi_minus) and
                          (self.bb_width[-1] > self.bb_width_sma[-1]) and
                          (entry_price > self.bb_upper[-1]))
            
            # Short Conditions ✅
            short_trigger = (crossover(self.vi_minus, self.vi_plus) and
                           (self.bb_width[-1] > self.bb_width_sma[-1]) and
                           (entry_price < self