```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaSqueezeStrategy(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_daily_risk = 0.03  # 3% max daily loss
    atr_period = 14
    bb_period = 20
    volume_sma_period = 20
    squeeze_lookback = 30

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])

        # Calculate indicators with self.I()
        self.upper_band = self.I(self._bb_upper, self.data.Close, name='Upper BB')
        self.middle_band = self.I(self._bb_middle, self.data.Close, name='Middle BB')
        self.lower_band = self.I(self._bb_lower, self.data.Close, name='Lower BB')
        
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period, name='Volume SMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # Calculate Bollinger Bandwidth
        self.bandwidth = self.I(self._calc_bandwidth, self.upper_band, self.lower_band, self.middle_band, name='Bandwidth')
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, self.squeeze_lookback, name='Min Bandwidth')

        # Initialize daily tracking
        self.current_day = None
        self.daily_start_equity = self._equity
        self.daily_max_loss = None

    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        return upper

    def _bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        return middle

    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        return lower

    def _calc_bandwidth(self, upper, lower, middle):
        return (upper - lower) / middle

    def next(self):
        # Moon Dev themed debug prints 🌙✨
        print(f"\n🌙 {self.data.index[-1]} - Close: {self.data.Close[-1]:.2f}")

        # Daily risk management check
        self._check_daily_risk()

        # Skip if daily loss limit reached
        if self.daily_start_equity - self.equity >= self.daily_max_loss:
            return

        # Check for volatility squeeze
        squeeze_signal = self.bandwidth[-1] <= self.min_bandwidth[-1]
        print(f"✨ Bandwidth: {self.bandwidth[-1]:.4f} | Min Bandwidth: {self.min_bandwidth[-1]:.4f}")

        # Entry logic
        if not self.position:
            if squeeze_signal:
                self._check_entries()

        # Exit logic and trailing stops
        else:
            self._check_exits()
            self._update_trailing_stops()

    def _check_daily_risk(self):
        current_date = self.data.index[-1].date()
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_start_equity = self.equity
            self.daily_max_loss = self.daily_start_equity * self.max_daily_risk
            print(f"\n🌕 NEW DAY: {current_date} | Starting Equity: {self.daily_start_equity:.2f} | Max Loss Allowed: {self.daily_max_loss:.2f}")

        if self.equity <= (self.daily_start_equity - self.daily_max_loss):
            print(f"🚨 DAILY LOSS LIMIT REACHED! Stopping trading for {current_date}")
            self.position.close()
            return

    def _check_entries(self):
        # Long