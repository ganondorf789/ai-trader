To address this problem, we need to create a backtesting implementation for the FractalBreakout strategy using the specified requirements. The strategy combines Heikin-Ashi candles, Bollinger Bands, Stochastic RSI, and fractal-based stops for entries, exits, and risk management.

### Approach
1. **Data Handling**: Load and preprocess the data to ensure proper column names and structure.
2. **Indicators Calculation**:
   - **Heikin-Ashi Candles**: Smoothed price action to filter market noise.
   - **Bollinger Bands**: Volatility indicator using the Heikin-Ashi close.
   - **Stochastic RSI**: Momentum indicator on resampled hourly data.
   - **Fractal Levels**: Recent swing highs/lows for dynamic stop placement.
3. **Entry/Exit Logic**: Conditions based on Heikin-Ashi candles, Bollinger Bands, and Stochastic RSI crossovers.
4. **Risk Management**: Position sizing based on risk percentage and trailing stops using fractal levels.

### Solution Code
```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

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
        
        # 1-hour Stochastic RSI
        def hourly_stochrsi(series):
            # Resample to hourly
            hourly = series.resample('1H').last().ffill()
            k, d = talib.STOCHRSI(hourly, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
            return k, d
        
        self.stoch_k, self.stoch_d = resample_apply('1H', hourly_stochrsi, self.data.Close)
        self.stoch_k = self.I(self.stoch_k.ffill(), name='Stoch K 1H')
        self.stoch_d = self.I(self.stoch_d.ffill(), name='Stoch D 1H')
        
        # Fractal levels (5-period window)
        self.fractal_high = self.I(talib.MAX, self.data.High, 5, name='Fractal High')
        self.fractal_low = self.I(talib.MIN, self.data.Low, 5, name='Fractal Low')
        
        print("🌙✨ Moon Dev Indicators Initialized! HA, BB, StochRSI, Fractals ready! 🚀")

    def next(self):
        if not self.position:
            # Long Entry Conditions
            ha_close_above_bb = self.ha_close[-1] > self.bb_upper[-1]
            stoch_cross_up = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2]
            
            if ha_close_above_