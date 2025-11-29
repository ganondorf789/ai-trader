Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing rules:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class OscilloFlow(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(self.data.High, self.data.Low, self.data.Close,
                                  fastk_period=14, slowk_period=3, slowk_matype=0,
                                  slowd_period=3, slowd_matype=0)
        self.slowk = self.I(lambda: slowk, name='%K')
        self.slowd = self.I(lambda: slowd, name='%D')
        
        # Chaikin Money Flow
        cmf_values = ta.cmf(self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=20)
        self.cmf = self.I(lambda: cmf_values, name='CMF')
        
        # Swing levels for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        
        print("🌙✨ OscilloFlow Strategy Initialized! Ready for launch! 🚀")

    def next(self):
        price = self.data.Close[-1]
        k, d = self.slowk[-1], self.slowd[-1]
        prev_k, prev_d = self.slowk[-2], self.slowd[-2]
        cmf = self.cmf[-1]
        
        # Crossover detection
        cross_up = k > d and prev_k <= prev_d
        cross_down = k < d and prev_k >= prev_d
        
        # Moon Dev debug prints
        print(f"🌙 Current K/D: {k:.1f}/{d:.1f} | CMF: {cmf:.2f} | Price: {price:.2f}")
        
        if not self.position:
            # Long entry logic
            if cross_up and k < 20 and d < 20 and cmf > 0:
                sl = self.swing_low[-1]
                risk_per_unit = price - sl
                if risk_per_unit > 0:
                    size = int(round((self.equity * self.risk_percent) / risk_per_unit))
                    tp = price + 2 * risk_per_unit
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🚀🌕 LONG SIGNAL! Size: {size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            # Short entry logic
            elif cross_down and k > 80 and d > 80 and cmf < 0:
                sl = self.swing_high[-1]
                risk_per_unit = sl - price
                if risk_per_unit > 0:
                    size = int(round((self.equity * self.risk_percent) / risk_per_unit))
                    tp = price - 2 * risk_per_unit
                    self.sell(size=size, sl=sl, tp=tp)
                    print(f"🌑🚀 SHORT SIGNAL! Size: {size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        
        else:
            # Long exit logic
            if self.position.is_long:
                if (cross_down and (k > 80 or d > 80)) or cmf < 0:
                    self.position.close()
                    print(f"✨🌙 Closing LONG at {price:.2f}")
            
            # Short exit logic
            elif self.position.is_short:
                if (cross_up and (k < 20 or d < 20)) or cmf >