I'll fix the code by removing all backtesting.lib imports and replacing any crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_concurrent_trades = 3
    
    def init(self):
        # Core indicators 🌙
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_width = self.I(lambda u, m, l: (u-l)/m, self.bb_upper, self.bb_middle, self.bb_lower)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.liq_long = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.liq_short = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)  # Fixed parameter name
        
        # Daily tracking ✨
        self.current_day = None
        self.daily_initial_equity = self._broker._cash

    def next(self):
        # Moon Dev Daily Reset 🌗
        today = self.data.index[-1].date()
        if today != self.current_day:
            self.current_day = today
            self.daily_initial_equity = self.equity
            print(f"🌙 New Moon Day: {today} | Initial Equity: ${self.daily_initial_equity:,.2f}")

        # Risk Forcefield 🛡️
        if (self.daily_initial_equity - self.equity)/self.daily_initial_equity >= 0.05:
            print(f"🚨 COSMIC LOSS LIMIT! {self.equity/self.daily_initial_equity-1:.2%} Daily Loss")
            return
            
        if len(self.trades) >= self.max_concurrent_trades:
            print("🌌 TOO MANY STARSHIPS IN HYPERSPACE!")
            return

        # Stellar Calculations 🌠
        price = self.data.Close[-1]
        bb_low_vol = self._is_bb_low_vol()
        
        # Long Entry Constellation 🌟
        long_liq_zone = self.liq_long[-1]
        if (bb_low_vol and
            price <= long_liq_zone * 1.005 and
            self._bullish_momentum()):
            
            sl = long_liq_zone - 1.5*self.atr[-1]
            tp = price + 2*(price - sl)
            self._execute_trade('LONG', price, sl, tp)

        # Short Entry Nebula 🌑
        short_liq_zone = self.liq_short[-1]
        if (bb_low_vol and
            price >= short_liq_zone * 0.995 and
            self._bearish_momentum()):
            
            sl = short_liq_zone + 1.5*self.atr[-1]
            tp = price - 2*(sl - price)
            self._execute_trade('SHORT',