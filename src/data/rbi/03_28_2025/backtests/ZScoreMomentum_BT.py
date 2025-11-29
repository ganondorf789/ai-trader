```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class ZScoreMomentumStrategy(Strategy):
    risk_percent = 0.01
    atr_period = 14
    bb_period = 20
    bb_std = 2
    zscore_lookback = 20
    volume_ma_period = 20

    def init(self):
        # Calculate indicators
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, 
                            self.data.Volume, 3, 10, name='Chaikin')
        
        # Z-Score calculation
        def zscore_func(series, timeperiod):
            mean = talib.SMA(series, timeperiod)
            std = talib.STDDEV(series, timeperiod)
            return (series - mean) / std
        self.zscore = self.I(zscore_func, self.chaokin, self.zscore_lookback, name='Z-Score')
        
        # Bollinger Bands on Z-Score
        self.bb_middle = self.I(talib.SMA, self.zscore, self.bb_period, name='BB_MID')
        self.bb_stddev = self.I(talib.STDDEV, self.zscore, self.bb_period, name='BB_STD')
        self.bb_upper = self.I(lambda: self.bb_middle + self.bb_std*self.bb_stddev, name='BB_UP')
        self.bb_lower = self.I(lambda: self.bb_middle - self.bb_std*self.bb_stddev, name='BB_LO')
        
        # Heikin-Ashi candles
        ha = ta.heikin_ashi(self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.ha_open = self.I(ha['HA_open'].values, name='HA_O')
        self.ha_high = self.I(ha['HA_high'].values, name='HA_H')
        self.ha_low = self.I(ha['HA_low'].values, name='HA_L')
        self.ha_close = self.I(ha['HA_close'].values, name='HA_C')
        
        # Additional indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                        self.atr_period, name='ATR')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='Vol_MA')

    def next(self):
        # Skip early bars and low volume periods
        if len(self.data) < 50 or self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # Get current values
        current_z, prev_z = self.zscore[-1], self.zscore[-2]
        current_upper, prev_upper = self.bb_upper[-1], self.bb_upper[-2]
        current_lower, prev_lower = self.bb_lower[-1], self.bb_lower[-2]
        
        # Heikin-Ashi values
        ha_close, ha_open = self.ha_close[-2:], self.ha_open[-2:]
        ha_low, ha_high = self.ha_low[-2:], self.ha_high[-2:]

        if not self.position:
            # Long entry logic
            if crossover(current_z, current_upper):
                ha_bullish = all(ha_close > ha_open)
                higher_lows = ha_low[-1] > ha_low[-2]
                if ha_bullish and higher_lows:
                    self.calculate_position(long=True)
            
            # Short entry logic
            elif crossunder(current_z, current_lower):
                ha_bearish = all(ha_close < ha_open)
                lower_highs = ha_high[-1] < ha_high[-2]
                if ha_bearish and lower_highs:
                    self.calculate_position(long=False)

        else:
            # Exit logic
            if self.position.is_long:
                if crossunder(current_z, current_upper) or (ha_close[-1] < ha_open[-1] and ha_close[-2] > ha_open[-2]):
                    self.position.close()
                    print(f"🌙✨💡 Closing LONG at {self.data.Close[-1]:.2f}")
            
            elif self.position.is_short:
                if crossover(current_z, current_lower) or (ha_close