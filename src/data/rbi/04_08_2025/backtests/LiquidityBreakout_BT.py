```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

# Data handling
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidityBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    bb_period = 20
    width_ma_period = 30
    swing_period = 20
    ck_period = 10
    ck_multiplier = 3
    
    def init(self):
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', output='upperband')
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_MIDDLE', output='middleband')
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', output='lowerband')
        
        # Bollinger Width calculations
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=self.width_ma_period, name='BB_WIDTH_SMA')
        
        # Swing High/Low (Liquidation Zones)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        
        # Volume Filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_SMA')
        
        # Chande Kroll Stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.ck_period, name='ATR')
        self.highest_high = self.I(talib.MAX, self.data.High, timeperiod=self.ck_period, name='HIGHEST_HIGH')
        self.lowest_low = self.I(talib.MIN, self.data.Low, timeperiod=self.ck_period, name='LOWEST_LOW')
        self.long_stop = self.highest_high - self.ck_multiplier * self.atr
        self.short_stop = self.lowest_low + self.ck_multiplier * self.atr
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready for launch! 🚀")

    def next(self):
        price = self.data.Close[-1]
        
        # Check for entries
        if not self.position:
            # Long Entry Conditions
            if (price > self.swing_high[-1] and
                self.bb_width[-1] < self.bb_width_sma[-1] and
                self.data.Volume[-1] > self.volume_sma[-1]):
                
                sl = self.swing_low[-1]
                risk_per_share = price - sl
                if risk_per_share <= 0:
                    return
                
                size = int(round((self.equity * self.risk_percent) / risk_per_share))
                if size > 0:
                    self.buy(size=size, sl=sl)
                    print(f"🚀🌙 LONG Signal! Price: {price:.2f}, Size: {size}, SL: {sl:.2f} | Moon Dev Power Activated!")
            
            # Short Entry Conditions
            elif (price < self.swing_low[-1