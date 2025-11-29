```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Ensure proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VoltaicRetest(Strategy):
    bb_period = 20
    bb_dev = 2
    bbw_threshold = 0.05
    volume_multiplier = 1.5
    atr_period = 14
    chandelier_atr_multiple = 3

    def init(self):
        # Bollinger Bands components
        def bb_upper(close, timeperiod, nbdevup, nbdevdn, matype):
            upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            return upper
        
        def bb_middle(close, timeperiod, nbdevup, nbdevdn, matype):
            _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            return middle
        
        def bb_lower(close, timeperiod, nbdevup, nbdevdn, matype):
            _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            return lower

        self.upper_band = self.I(bb_upper, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, 0, name='UpperBB')
        self.middle_band = self.I(bb_middle, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, 0, name='MiddleBB')
        self.lower_band = self.I(bb_lower, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, 0, name='LowerBB')

        # Bollinger Bandwidth
        self.bbw = self.I(lambda u, m, l: (u - l)/m, self.upper_band, self.middle_band, self.lower_band, name='BBW')

        # Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='VolumeSMA')

        # ATR for volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')

        # State variables
        self.breakout_direction = None
        self.breakout_level = None
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Moon Dev state reset 🌙
        if self.bbw[-1] >= self.bbw_threshold:
            self.breakout_direction = None

        if not self.position:
            # Volatility contraction detection
            if len(self.bbw) > 1 and self.bbw[-1] < self.bbw_threshold and self.bbw[-2] >= self.bbw_threshold:
                print(f"🌙 MOON DEV ALERT: Volatility contraction detected! BBW={self.bbw[-1]:.4f}")

            # Breakout confirmation
            if self.bbw[-1] < self.bbw_threshold:
                vol_spike = self.data.Volume[-1] > self.volume_multiplier * self.volume_sma[-1]
                close_above_upper = self.data.Close[-1] > self.upper_band[-1]
                close_below_lower = self.data.Close[-1] < self.lower_band[-1]

                if vol_spike and close_above_upper:
                    self.breakout_direction = 'long'
                    self.breakout_level = self.upper