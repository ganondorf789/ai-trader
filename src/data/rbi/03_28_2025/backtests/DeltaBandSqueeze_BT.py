```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev Themed Strategy 🌙✨
class DeltaBandSqueeze(Strategy):
    # Strategy Parameters
    bb_period = 20
    bb_dev = 2
    delta_period = 3
    atr_period = 14
    width_median_period = 20
    risk_percent = 0.01
    max_bars_held = 5

    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # Calculate Bollinger Bands 🌈
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, 
            nbdevdn=self.bb_dev, 
            matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Calculate Bollinger Band Width 🔍
        bb_width = self.I(lambda: self.bb_upper - self.bb_lower, name='BB_Width')
        self.bb_width_sma = self.I(talib.SMA, bb_width, self.bb_period, name='BB_Width_SMA')
        self.bb_width_median = self.I(talib.MEDIAN, bb_width, self.width_median_period, name='BB_Width_Median')

        # Calculate Volume Delta 🌊
        delta = np.where(self.data.Close > pd.Series(self.data.Close).shift(1), 
                       self.data.Volume, 
                       -self.data.Volume)
        self.delta_sum = self.I(talib.SUM, delta, self.delta_period, name='Delta_Sum')

        # Calculate ATR for stops 🔐
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')

        # Delta Spike Detection ⚡
        self.delta_mean = self.I(talib.SMA, self.delta_sum, 20, name='Delta_Mean')
        self.delta_std = self.I(talib.STDDEV, self.delta_sum, 20, name='Delta_Std')

        # VIX Check (if available) 📈
        self.vix = self.data.df['vix'] if 'vix' in self.data.df.columns else None
        if self.vix is None:
            print("🌙✨ Moon Dev Warning: VIX data not found - trading without volatility filter")

        # Trackers for risk management 🛡️
        self.max_equity = self._broker.starting_cash
        self.entry_bar = 0

    def next(self):
        # Moon Dev Risk Management First! 🌙
        current_equity = self._broker.equity
        self.max_equity = max(self.max_equity, current_equity)
        drawdown = (self.max_equity - current_equity) / self.max_equity
        
        if drawdown >= 0.05:
            if self.position:
                self.position.close()
                print("🚨🌙 MAX DRAWDOWN REACHED! Closing all positions")
            return

        # Entry Logic 🌌
        if not self.position:
            # Core Conditions
            price_condition = self.data.Close[-1] < self.bb_lower[-1]
            volatility_condition = (self.bb_upper[-1] - self.bb_lower[-1]) < self.bb_width_sma[-1]
            delta_condition = (self.delta_sum[-1] < self.delta_sum[-2] < self.delta_sum[-3]) and (self.delta_sum[-1] < 0)
            vix_condition = (self.vix[-1] > 15) if self.vix is not None else True

            if all([price_condition, volatility_condition, delta_condition, vix_condition]):
                # Calculate Position Size 💰
                risk_amount = current_equity * self.risk_percent
                stop_distance = 2 * self.atr[-1]
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))

                if position_size > 0:
                    self.sell(size=position_size, 
                             sl=self.data.Close[-1] + stop_distance,
                             tag="🌙🚀 Short Entry")
                    self.entry_bar = len