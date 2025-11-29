```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class SqueezeSurge(Strategy):
    risk_percent = 0.01  # 1% of equity per trade 🌙
    
    def init(self):
        # Bollinger Bands components ✨
        def upper_bb(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        self.upper_bb = self.I(upper_bb, self.data.Close, name='UpperBB')
        
        def lower_bb(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return lower
        self.lower_bb = self.I(lower_bb, self.data.Close, name='LowerBB')
        
        # Bandwidth calculations 🌀
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_bb, self.lower_bb, name='Bandwidth')
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=20, name='MinBandwidth')
        
        # Volume and trend indicators 📊
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolumeSMA20')
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA200')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
        # Trailing stop tracking 🌙
        self.trailing_stop = None
        self.highest_high = None
        self.lowest_low = None

    def next(self):
        # Exit logic first ✨
        if self.position:
            atr_value = self.atr[-1]
            if self.position.is_long:
                self.highest_high = max(self.highest_high, self.data.High[-1])
                self.trailing_stop = self.highest_high - 3 * atr_value
                if self.data.Close[-1] <= self.trailing_stop:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Long closed at {self.data.Close[-1]:.2f} | Trail Stop {self.trailing_stop:.2f} ✨")
            else:
                self.lowest_low = min(self.lowest_low, self.data.Low[-1])
                self.trailing_stop = self.lowest_low + 3 * atr_value
                if self.data.Close[-1] >= self.trailing_stop:
                    self.position.close()
                    print(f"🌑 Moon Dev Exit: Short closed at {self.data.Close[-1]:.2f} | Trail Stop {self.trailing_stop:.2f} ✨")
            return

        # Entry logic requires 200 periods 📆
        if len(self.data) < 200:
            return

        # Strategy conditions check ✅
        squeeze = self.bandwidth[-1] <= self.min_bandwidth[-1]
        volume_surge = self.data.Volume[-1] >= 1.5 * self.volume_sma20[-1]
        price_below_ma = self.data.Close[-1] < self.sma200[-1]
        breakout_long = self.data.Close[-1] > self.upper_bb[-1]
        breakout_short = self.data.Close[-1] < self.lower_bb[-1]

        if all([squeeze, volume_surge, price_below_ma]):
            atr_value = self.atr[-1]
            if atr_value <= 0:
                return  # Safety check ⚠️