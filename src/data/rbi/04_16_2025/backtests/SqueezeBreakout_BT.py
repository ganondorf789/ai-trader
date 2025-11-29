Here's the complete backtesting.py implementation for the SqueezeBreakout strategy:

```python
# 🌙 Moon Dev SqueezeBreakout Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class SqueezeBreakout(Strategy):
    risk_percentage = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Calculate all indicators upfront using TA-Lib
        close = self.data.Close.values.astype(float)
        high = self.data.High.values.astype(float)
        low = self.data.Low.values.astype(float)
        
        # Bollinger Bands (20,2)
        self.bb_upper, _, self.bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.I(lambda: self.bb_upper, name='BB_upper')
        self.I(lambda: self.bb_lower, name='BB_lower')
        
        # Donchian Channel (20)
        self.donchian_high = talib.MAX(high, timeperiod=20)
        self.donchian_low = talib.MIN(low, timeperiod=20)
        self.I(lambda: self.donchian_high, name='Donchian_high')
        self.I(lambda: self.donchian_low, name='Donchian_low')
        
        # ADX and ATR (14)
        self.adx = talib.ADX(high, low, close, timeperiod=14)
        self.atr = talib.ATR(high, low, close, timeperiod=14)
        self.I(lambda: self.adx, name='ADX')
        self.I(lambda: self.atr, name='ATR')
        
        # Volatility calculations
        bb_width = self.bb_upper - self.bb_lower
        self.bb_width_min = talib.MIN(bb_width, 20)
        self.I(lambda: bb_width, name='BB_width')
        self.I(lambda: self.bb_width_min, name='BB_width_min')
        
        print("🌙✨ Moon Dev Strategy Activated! Ready for Launch 🚀")

    def next(self):
        if len(self.data) < 20:  # Wait for all indicators to form
            return

        # Current market conditions 🌙
        price = self.data.Close[-1]
        adx = self.data.ADX[-1]
        atr = self.data.ATR[-1]
        
        # Squeeze detection
        in_squeeze = (self.data.BB_width[-1] <= self.data.BB_width_min[-1])
        
        # Donchian expansion check
        donchian_width = self.data.Donchian_high[-1] - self.data.Donchian_low[-1]
        prev_width = self.data.Donchian_high[-2] - self.data.Donchian_low[-2] if len(self.data) > 1 else 0
        expanding = donchian_width > prev_width

        # Entry logic 🚀
        if not self.position:
            # Long entry conditions
            long_cond = (price > self.data.BB_upper[-1] and
                         price > self.data.Donchian_high[-1] and
                         expanding and
                         in_squeeze)
            
            # Short entry conditions
            short_cond = (price < self.data.BB_lower[-1] and
                          price < self.data.Donchian_low[-1] and
                          expanding and
                          in_squeeze)

            if long_cond:
                risk_amount = self.equity * self.risk_percentage
                sl_distance = 2 * atr
                position_size = int(round(risk_amount / sl_distance))
                if position_size > 0:
                    self.buy(size=position_size, sl=price - sl_distance)
                    print(f"🚀🌙