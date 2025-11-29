Here's the fully debugged code with Moon Dev themed improvements and proper position sizing:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and prepare data with Moon Dev cosmic energy 🌌
print("🌙 Moon Dev Debug: Loading cosmic market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns with lunar precision
print("🌙 Moon Dev Debug: Purifying data with moonbeams...")
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

class VoltaicSurge(Strategy):
    bb_period = 20
    bb_dev = 2
    kc_period = 20
    kc_dev = 1.5
    bbw_window = 960  # 10 days in 15m intervals
    volume_ma_window = 960
    volume_multiplier = 2
    exit_bars = 5
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        print("🌙 Moon Dev Debug: Initializing celestial indicators...")
        # Bollinger Bands
        self.upper, self.mid, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                 timeperiod=self.bb_period, 
                                                 nbdevup=self.bb_dev, 
                                                 nbdevdn=self.bb_dev)
        
        # BBW Calculation
        self.bbw = (self.upper - self.lower) / self.mid
        self.bbw_percentile = self.I(self._rolling_percentile, self.bbw, self.bbw_window, 20)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_window)
        
        # Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, self.kc_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.kc_period)
        self.kc_upper = self.I(lambda: self.ema + self.kc_dev * self.atr)
        self.kc_lower = self.I(lambda: self.ema - self.kc_dev * self.atr)
        
        self.entry_bar = 0

    def _rolling_percentile(self, series, window, percentile):
        return series.rolling(window).apply(lambda x: np.percentile(x, percentile))
    
    def next(self):
        price = self.data.Close[-1]
        
        if self.position:
            # Exit logic
            if self.position.is_long:
                if price <= self.kc_lower[-1] or len(self.data) - self.entry_bar >= self.exit_bars:
                    self.position.close()
                    print(f"🌙 Moon Exit: LONG closed at {price:.2f} {'📉' if price < self.position.entry_price else '💰'}")
            else:
                if price >= self.kc_upper[-1] or len(self.data) - self.entry_bar >= self.exit_bars:
                    self.position.close()
                    print(f"🌙 Moon Exit: SHORT closed at {price:.2f} {'📈' if price > self.position.entry_price else '💰'}")
        else:
            # Entry logic
            bb_contraction = self.bbw_percentile[-1] < 20
            volume_surge = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_multiplier
            
            if bb_contraction and volume_surge:
                if price > self.upper[-1]:
                    # Long entry
                    sl = self.lower[-1]
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / (price - sl)))
                    if position_size > 0:  # Ensure valid position size
                        self.buy(size=position_size, sl=sl)
                        self.entry_bar = len(self.data)
                        print(f"🚀 Moon LONG Entry: {price:.2f} | Size: {