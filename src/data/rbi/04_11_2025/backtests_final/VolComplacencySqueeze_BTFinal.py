Here's the debugged code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# Data preparation - Moon Dev cosmic data alignment ✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index('datetime')

class VolComplacencySqueeze(Strategy):
    risk_percent = 0.01  # Risk 1% of equity per trade
    tp_multiplier = 2    # 2:1 reward:risk ratio
    lookback_period = 100  # For percentile calculations
    skew_window = 20     # Window for skew calculation
    
    def init(self):
        # Bollinger Bands components - Cosmic volatility measurement 🌌
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
            return (upper - lower) / middle
        
        self.bb_width = self.I(bb_width, self.data.Close)
        self.bb_percentile = self.I(ta.percentile, self.bb_width, length=self.lookback_period)
        
        # Volume-adjusted skew - Moon Dev Sentiment Scanner 🌙
        returns = self.I(lambda c: pd.Series(c).pct_change(), self.data.Close)
        skew = self.I(ta.skew, returns, length=self.skew_window)
        
        vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        vol_std = self.I(talib.STDDEV, self.data.Volume, 20)
        vol_z = self.I(lambda v, sma, std: (v - sma)/std, 
                      self.data.Volume, vol_sma, vol_std)
        
        self.vol_skew = self.I(lambda s, vz: s * vz, skew, vol_z)
        
        # Track entry conditions - Moon Position Tracker 🌑
        self.entry_width = None
        self.entry_price = None

    def next(self):
        price = self.data.Close[-1]
        
        # Entry logic - Cosmic alignment detection ✨
        if not self.position:
            if (self.bb_percentile[-1] < 20 and 
                self.vol_skew[-1] > np.nanmean(self.vol_skew[-self.lookback_period:]) + 
                np.nanstd(self.vol_skew[-self.lookback_period:])):
                
                # Risk management - Moon Safety Protocols 🚀
                risk_amount = self.equity * self.risk_percent
                atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 14)[-1]
                stop_price = price + 1.5 * atr
                position_size = risk_amount / (stop_price - price)
                position_size = int(round(position_size))  # Ensure whole units
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_price, 
                             tp=price - (stop_price - price)*self.tp_multiplier)
                    self.entry_width = self.bb_width[-1]
                    self.entry_price = price
                    print(f"🌙✨ MOON DEV ENTRY: Short Volatility Activated! ✨")
                    print(f"Entry Price: {price:.2f} | Size: {position_size}")
                    print(f"🌌 Volatility Percentile: {self.bb_percentile[-1]:.2f}%")
                    print(f"🌀 Volume-Skew Z-Score: {self.vol_skew[-1]:.2f}")

        # Exit logic - Cosmic exit signals 🌕
        elif self.position:
            # Volatility spike exit
            if self.bb_percentile[-1] > 50:
                self.position.close()
                print(f"🚨 MOON DEV EXIT: Volatility Expansion Detected!")
                print(f"🌕 Current Percentile: {self.bb_percentile[-1]:.2f}%")
                print(f"💎 P