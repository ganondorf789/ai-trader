Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime']
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
data.set_index('datetime', inplace=True)

class BandwidthBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands indicators
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[2], self.data.Close)
        self.bb_width = self.I(lambda u,l: u-l, self.bb_upper, self.bb_lower)
        self.bb_width_sma10 = self.I(talib.SMA, self.bb_width, timeperiod=10)
        
        # Volume indicator
        self.volume_median = self.I(ta.median, self.data.Volume, length=50)
        
        # Keltner Channel components
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.ema20 = self.I(talib.EMA, typical_price, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        
    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Current BB Width: {self.bb_width[-1]:.2f} vs SMA10: {self.bb_width_sma10[-1]:.2f}")
            print(f"📈 Volume: {self.data.Volume[-1]:.2f} vs Median: {self.volume_median[-1]:.2f}")
            print(f"✨ KC Range: {self.ema20[-1]:.2f} ± {1.5*self.atr20[-1]:.2f}\n")
        
        if not self.position:
            contraction = self.bb_width[-1] < self.bb_width_sma10[-1]
            volume_ok = self.data.Volume[-1] > self.volume_median[-1]
            
            # Long entry logic
            if contraction and volume_ok and self.data.Close[-1] > self.bb_upper[-1]:
                risk_amount = self.equity * self.risk_per_trade
                entry_price = self.data.Close[-1]
                kc_lower = self.ema20[-1] - 1.5*self.atr20[-1]
                risk_per_unit = entry_price - kc_lower
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        tp_price = entry_price + 2*risk_per_unit
                        self.buy(size=size, sl=kc_lower, tp=tp_price)
                        print(f"🚀🌙 LONG ENTRY @ {entry_price:.2f}")
                        print(f"   ░SL: {kc_lower:.2f} | TP: {tp_price:.2f} | Size: {size}\n")
            
            # Short entry logic        
            elif contraction and volume_ok and self.data.Close[-1] < self.bb_lower[-1]:
                risk_amount = self.equity * self.risk_per_trade
                entry_price = self.data.Close[-1]
                kc_upper = self.ema20[-1] + 1.5*self.atr20[-1]
                risk_per_unit = kc_upper - entry_price
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        tp_price = entry_price - 2*risk_per_unit