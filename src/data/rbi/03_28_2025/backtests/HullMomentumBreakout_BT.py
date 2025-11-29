```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index(pd.to_datetime(data['datetime']))

class HullMomentumBreakout(Strategy):
    hma_length = 14
    cmo_period = 20
    swing_window = 20
    vol_ma_period = 20
    risk_pct = 0.01
    min_slope_angle = 30
    cmo_threshold = 50
    
    def init(self):
        # Core Indicators 🌙
        self.hma = self.I(ta.hma, self.data.Close, length=self.hma_length, name='HMA')
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=self.cmo_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        
    def next(self):
        # Moon Dev Debug Prints ✨
        print(f"\n🌙 New Candle: {self.data.index[-1]} | Price: {self.data.Close[-1]:.2f}")
        print(f"HMA Slope: {self._calculate_hma_slope():.2f}° | CMO: {self.cmo[-1]:.2f}")
        
        # Manage open positions 🚀
        for trade in self.trades:
            if self._should_exit(trade):
                trade.close()
                print(f"🌙 Closing Position: {trade.tag}")
        
        # Entry logic when flat 🌗
        if not self.position:
            self._check_entries()
            
    def _calculate_hma_slope(self):
        """Calculate HMA slope angle in degrees 🌙"""
        if len(self.hma) < 3:
            return 0
        delta = self.hma[-1] - self.hma[-3]  # 2-period slope
        return np.degrees(np.arctan(delta/self.hma[-1]))

    def _check_entries(self):
        """Moon Entry Logic 🌌"""
        hma_slope = self._calculate_hma_slope()
        vol_condition = self.data.Volume[-1] > self.vol_ma[-1]
        
        # Long Setup 🌟
        if (hma_slope > self.min_slope_angle and
            self.cmo[-1] > self.cmo_threshold and
            self.data.Close[-1] > self.swing_high[-1] and
            vol_condition):
            
            self._enter_long()
            
        # Short Setup 🌑
        elif (hma_slope < -self.min_slope_angle and
              self.cmo[-1] < -self.cmo_threshold and
              self.data.Close[-1] < self.swing_low[-1] and
              vol_condition):
            
            self._enter_short()

    def _enter_long(self):
        """Execute Bullish Trade 🌟"""
        sl = self.swing_low[-1]
        risk = self.data.Close[-1] - sl
        size = self._calculate_position_size(risk)
        
        # Fibonacci Targets 🌉
        fib_ext = self.data.Close[-1] + risk * 1.618
        fib_ret = self.data.Close[-1] - risk * 0.382 * (self.data.Volume[-1]/self.vol_ma[-1])
        
        self.buy(size=size,
                 sl=sl,
                 tp=fib_ext,
                 tag=f"🌕 LONG | TP: {fib_ext:.2f}")
        print(f"🚀 BULLISH BREAKOUT! Size: {size} | SL: {sl:.2f}")

    def _enter_short(self):
        """Execute Bearish Trade 🌑"""
        sl = self