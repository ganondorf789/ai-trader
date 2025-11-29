I'll fix the code by removing all backtesting.lib imports and replacing any crossover functions with manual comparisons. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's KeltnerDivergence Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# 🚀 Data Loading & Cleaning
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

class KeltnerDivergence(Strategy):
    ema_period = 20
    atr_multiplier = 2
    cmf_period = 20
    swing_window = 5
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        # 🌗 Core Indicators
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.ema_period)
        self.upper_kc = self.I(lambda: self.ema + self.atr * self.atr_multiplier, name='Upper KC')
        self.lower_kc = self.I(lambda: self.ema - self.atr * self.atr_multiplier, name='Lower KC')
        
        # 💰 CMF Calculation
        cmf_values = ta.cmf(self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=self.cmf_period)
        self.cmf = self.I(lambda: cmf_values, name='CMF')
        
        # 🐂🐻 Elder-Ray Index
        self.bull_power = self.I(lambda: self.data.High - self.ema, name='Bull Power')
        self.bear_power = self.I(lambda: self.data.Low - self.ema, name='Bear Power')
        
        # 🏔️ Swing Detection
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='Swing Low')
        
        # 📈 Track swing extremes
        self.last_swing_highs = []
        self.last_swing_lows = []

    def calculate_size(self, risk_per_unit):
        """✨ Calculate position size based on risk percentage"""
        risk_amount = self.equity * self.risk_pct
        return risk_amount / risk_per_unit if risk_per_unit > 0 else 0

    def next(self):
        # 🌙 Update swing history
        if self.data.High[-1] == self.swing_high[-1]:
            self.last_swing_highs.append((self.data.High[-1], self.cmf[-1]))
            if len(self.last_swing_highs) > 2:
                self.last_swing_highs.pop(0)
                
        if self.data.Low[-1] == self.swing_low[-1]:
            self.last_swing_lows.append((self.data.Low[-1], self.cmf[-1]))
            if len(self.last_swing_lows) > 2:
                self.last_swing_lows.pop(0)

        # ✨ Entry Logic
        if not self.position:
            # 🚀 Long Entry
            if (self.data.Close[-1] > self.upper_kc[-1] and 
                len(self.last_swing_highs) >= 2 and
                self.last_swing_highs[-1][0] > self.last_swing_highs[-2][0] and
                self.last_swing_highs[-1][1] < self.last_swing_highs[-2][1]):
                
                sl_price = min(self.swing_low[-1], self.lower_kc[-1])
                risk_per_unit = self.data.Close[-1] - sl_price
                position_size = self.calculate_size(risk_per_unit)
                if position_size > 0:
                    self.buy(size