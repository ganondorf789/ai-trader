I'll fix the code by removing the `backtesting.lib` import and replacing any usage of its functions. Here's the corrected version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import datetime
from collections import defaultdict

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSyncChannel(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙 Risk Management
    
    def init(self):
        # Core indicators ✨
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20 🌙')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR20 ⚡')
        self.keltner_upper = self.I(lambda: self.ema20 + 2.5*self.atr20, name='Keltner Upper 🚀')
        self.keltner_lower = self.I(lambda: self.ema20 - 2.5*self.atr20, name='Keltner Lower 🛑')
        
        # VWAP system ⚖️
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP ⚖️')
        self.vwap_dev = self.I(lambda: self.data.Close - self.vwap, name='VWAP Dev 📏')
        self.vwap_std = self.I(talib.STDDEV, self.vwap_dev, timeperiod=20, name='VWAP STD 🔍')
        
        # Volume filter 🔊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Vol MA 🔊')
        
        # Risk management 🛡️
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14 🛡️')
        
        # Trade tracking 📊
        self.daily_trades = defaultdict(int)
        
    def next(self):
        current_time = self.data.index[-1].time()
        today = self.data.index[-1].date()
        
        # Session filter 🌙🌃
        if (datetime.time(0, 0) <= current_time <= datetime.time(0, 30)) or current_time >= datetime.time(23, 30):
            print(f"🌙 Night Owl Protection: Skipping trading at {current_time}")
            return
        
        if not self.position:
            # Trade counter check
            if self.daily_trades[today] >= 3:
                print(f"🌙 Daily Trade Limit Reached! ({today})")
                return
            
            # Long conditions ✨
            long_trigger = (
                self.data.Close[-1] > self.keltner_upper[-1] and
                self.vwap_dev[-1] < 0.5*self.vwap_std[-1] and
                self.data.Volume[-1] > self.vol_ma[-1]
            )
            
            # Short conditions 🎯
            short_trigger = (
                self.data.Close[-1] < self.keltner_lower[-1] and
                self.vwap_dev[-1] > -0.5*self.vwap_std[-1] and
                self.data.Volume[-1] > self.vol_ma[-1]
            )
            
            if long_trigger or short_trigger:
                atr = self.atr14[-1]
                risk_amount = self.equity * self.risk_per_trade
                
                if long_trigger:
                    sl = self.data.Close[-1] - atr
                    risk_per_unit = self.data.Close[-1] - sl
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        self.buy(size=size