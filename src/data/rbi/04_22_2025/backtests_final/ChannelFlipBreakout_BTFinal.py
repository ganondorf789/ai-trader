```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import pandas_ta as pta

class ChannelFlipBreakout(Strategy):
    def init(self):
        data = self.data
        
        # Calculate Donchian Channels (20-period)
        self.donchian_upper = self.I(talib.MAX, data.High, timeperiod=20)
        self.donchian_lower = self.I(talib.MIN, data.Low, timeperiod=20)
        
        # Calculate SuperTrend (10-period ATR, 3x multiplier)
        def supertrend_wrapper(high, low, close):
            supertrend = pta.supertrend(high, low, close, length=10, multiplier=3)
            return supertrend['SUPERTd_10_3.0']
        self.supertrend = self.I(supertrend_wrapper, data.High, data.Low, data.Close)
        
        # Calculate ATR (14-period)
        self.atr = self.I(talib.ATR, data.High, data.Low, data.Close, timeperiod=14)
        
        # Median ATR filter (50-period)
        self.atr_median = self.I(talib.MEDPRICE, self.atr, timeperiod=50)
        
        self.prev_supertrend = None

    def next(self):
        price = self.data.Close[-1]
        atr_value = self.atr[-1]
        equity = self.equity
        
        # Moon Dev debug variables
        moon_emoji = "🌙✨"
        long_signal = "🚀 LONG ALERT" 
        short_signal = "🐻 SHORT ALERT"
        
        # Entry/Exit conditions
        long_entry = (
            price > self.donchian_upper[-1] and
            self.supertrend[-2] == -1 and
            self.supertrend[-1] == 1 and
            atr_value > self.atr_median[-1]
        )
        
        short_entry = (
            price < self.donchian_lower[-1] and
            self.supertrend[-2] == 1 and
            self.supertrend[-1] == -1 and
            atr_value > self.atr_median[-1]
        )
        
        # Risk management calculations
        risk_percent = 0.02
        stop_distance = atr_value * 1
        position_size = int(round((equity * risk_percent) / stop_distance))
        
        # Trade execution logic
        if not self.position:
            if long_entry:
                sl = price - stop_distance * 2  # 2x ATR trail
                print(f"{moon_emoji} {long_signal} BUY @ {price:.2f}, Size: {position_size}")
                self.buy(size=position_size, sl=sl)
                
            elif short_entry:
                sl = price + stop_distance * 2  # 2x ATR trail
                print(f"{moon_emoji} {short_signal} SELL @ {price:.2f}, Size: {position_size}")
                self.sell(size=position_size, sl=sl)
        
        # Trailing stop logic
        elif self.position.is_long:
            trail_price = self.data.High[-1] - (atr_value * 2)
            self.position.sl = max(self.position.sl, trail_price)
            
            if self.supertrend[-1] == -1:
                print(f"{moon_emoji} 🌧️ SUPER TREND REVERSAL EXIT LONG @ {price:.2f}")
                self.position.close()
                
        elif self.position.is_short:
            trail_price = self.data.Low[-1] + (atr_value * 2)
            self.position.sl = min(self.position.sl, trail_price)
            
            if self.supertrend[-1] == 1:
                print(f"{moon_emoji} 🌞 SUPER TREND REVERSAL EXIT SHORT @ {price:.2f}")
                self.position.close()

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert to datetime