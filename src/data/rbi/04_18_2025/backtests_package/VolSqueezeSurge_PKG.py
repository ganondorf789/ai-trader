Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class VolSqueezeSurge(Strategy):
    bb_period = 20
    bb_dev = 2
    vol_ma_period = 20
    hist_percentile_window = 100
    percentile_threshold = 20
    volume_multiplier = 1.5
    risk_per_trade = 0.01
    max_bars_held = 5

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)

        # Calculate indicators using TA-Lib
        close = self.data.Close
        volume = self.data.Volume
        
        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, 
                                                    timeperiod=self.bb_period,
                                                    nbdevup=self.bb_dev,
                                                    nbdevdn=self.bb_dev,
                                                    matype=0, name='BBANDS')
        
        # BB Width calculation
        self.bb_width = (self.upper - self.lower) / self.middle * 100
        self.bb_width_percentile = self.I(lambda x: x.rolling(self.hist_percentile_window).quantile(self.percentile_threshold/100),
                                       self.bb_width, name='BB_PERCENTILE')
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name='VOL_MA')
        
        print("🌙 Moon Dev Indicators Initialized! BBANDS, Volume MA, Percentile Ready 🚀")

    def next(self):
        current_idx = len(self.data)-1
        if current_idx < self.hist_percentile_window:
            return

        # Entry conditions
        if not self.position:
            # Volatility squeeze condition
            squeeze_condition = (self.bb_width[-1] < self.bb_width_percentile[-1])
            
            # Volume surge validation
            volume_condition = (self.data.Volume[-1] > self.volume_multiplier * self.vol_ma[-1])
            
            if squeeze_condition and volume_condition:
                # Determine breakout direction
                if self.data.Close[-1] > self.upper[-1]:
                    self.enter_long()
                elif self.data.Close[-1] < self.lower[-1]:
                    self.enter_short()

        # Exit conditions
        else:
            bars_in_trade = current_idx - self.position.entry_bar
            if bars_in_trade >= self.max_bars_held:
                self.position.close()
                print(f"🌙 Moon Dev Time Exit: Held {bars_in_trade+1} bars ✨")

    def enter_long(self):
        entry_price = self.data.Open[-1]
        bb_width_val = self.bb_width[-1]
        
        # Risk calculations
        stop_pct = 1.5 * bb_width_val / 100
        take_profit_pct = 2 * bb_width_val / 100
        
        stop_price = entry_price * (1 - stop_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = entry_price - stop_price
        position_size = risk_amount / risk_per_share
        position_size = int(round(position_size))
        
        if position_size > 0:
            self.buy(size=position_size, 
                    sl=stop_price,
                    tp=take_profit_price)
            print(f"🌙🚀 LONG Entry! {position_size} shares at {entry_price:.2f} | SL: {stop_price:.2f} | TP: {take_profit_price:.2f}")

    def enter_short(self):
        entry_price = self.data.Open[-1]
        bb_width_val = self.bb_width[-1]
        
        # Risk calculations
        stop_pct = 1.5 * bb_width_val / 100
        take_profit_pct = 2 * bb_width_val / 100
        
        stop_price = entry_price * (1 + stop_pct)
        take_profit_price = entry_price * (1 - take_profit_pct)
        
        # Position sizing
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share =