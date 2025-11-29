I'll fix the code to remove any `backtesting.lib` imports and replace them with appropriate alternatives. Here's the corrected version:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy

class FractalBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade

    def init(self):
        # Clean and prepare Heikin-Ashi data
        ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        ha_open = pd.Series(index=self.data.index, dtype=float)
        ha_open.iloc[0] = (self.data.Open.iloc[0] + self.data.Close.iloc[0]) / 2
        for i in range(1, len(self.data)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_high = pd.concat([self.data.High, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([self.data.Low, ha_open, ha_close], axis=1).min(axis=1)
        
        # Add Heikin-Ashi indicators
        self.ha_close = self.I(lambda: ha_close, name='HA Close')
        self.ha_open = self.I(lambda: ha_open, name='HA Open')
        self.ha_high = self.I(lambda: ha_high, name='HA High')
        self.ha_low = self.I(lambda: ha_low, name='HA Low')
        
        # Bollinger Bands on HA Close
        self.bb_upper = self.I(lambda: talib.BBANDS(ha_close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], name='BB Upper')
        self.bb_lower = self.I(lambda: talib.BBANDS(ha_close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], name='BB Lower')
        
        # 1-hour Stochastic RSI - Custom implementation without backtesting.lib
        def resample_to_hourly(series):
            # Create a DataFrame with the series and its index
            df = pd.DataFrame({'value': series}, index=self.data.index)
            # Resample to hourly and take last value
            hourly = df.resample('1H').last()
            return hourly['value'].ffill()
        
        def calculate_stochrsi(series):
            # Calculate StochRSI using TA-Lib
            k, d = talib.STOCHRSI(series, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
            return k, d
        
        # Resample Close to hourly
        hourly_close = resample_to_hourly(self.data.Close)
        # Calculate StochRSI on hourly data
        stoch_k, stoch_d = calculate_stochrsi(hourly_close)
        # Forward fill to original timeframe
        stoch_k = stoch_k.reindex(self.data.index).ffill()
        stoch_d = stoch_d.reindex(self.data.index).ffill()
        
        self.stoch_k = self.I(lambda: stoch_k, name='Stoch K 1H')
        self.stoch_d = self.I(lambda: stoch_d, name='Stoch D 1H')
        
        # Fractal levels (5-period window)
        self.fractal_high = self.I(talib.MAX, self.data.High, 5, name='Fractal High')
        self.fractal_low = self.I(talib.MIN, self.data.Low, 5, name='Fractal Low')
        
        print("🌙✨ Moon Dev Indicators Initialized! HA, BB, StochRSI, Fractals ready! 🚀")

    def next(self):
        if not self.position:
            # Long Entry Conditions
            ha_close_above_bb = self.ha_close[-1] > self.bb_upper[-1]
            stoch_cross_up = self.stoch_k[-2] <= self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1]
            
            if ha_close_above_bb and stoch_cross_up:
                # Calculate position size based on risk
                stop_loss = self.fractal_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                position_size = risk_amount / (self.data.Close[-1] - stop_loss)
                
                self.buy(size=position_size