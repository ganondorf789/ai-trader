Here's the fixed code with all backtesting.lib imports removed and replaced with proper Moon Dev themed implementations:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
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

class VolumetricStochastic(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙💰
    max_open_positions = 3
    atr_multiplier_tp = 1.5
    atr_multiplier_sl = 0.75
    stochastic_period = 14
    swing_period = 20

    def init(self):
        # Precompute indicators using TA-Lib 🌙📊
        h, l, c = self.data.High, self.data.Low, self.data.Close
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(h, l, c,
                                 fastk_period=14,
                                 slowk_period=3,
                                 slowk_matype=0,
                                 slowd_period=3,
                                 slowd_matype=0)
        self.slowk = self.I(lambda: slowk, name='SlowK')
        self.slowd = self.I(lambda: slowd, name='SlowD')

        # On-Balance Volume
        self.obv = self.I(talib.OBV, c, self.data.Volume)
        
        # ATR and Swing Points
        self.atr = self.I(talib.ATR, h, l, c, 14)
        self.swing_high = self.I(talib.MAX, h, self.swing_period)
        self.swing_low = self.I(talib.MIN, l, self.swing_period)

    def next(self):
        # Moon Dev Time Filter 🌙⏰
        if self.data.index[-1].hour >= 6:
            return

        # Position Management 🌙🔢
        if len(self.trades) >= self.max_open_positions:
            return

        current_close = self.data.Close[-1]
        atr = self.atr[-1]

        # Moon Dev Long Entry Logic 🌙📈
        if ((self.slowk[-2] <= 20 and self.slowk[-1] > 20) and  # Replaced crossabove
            self.slowk[-1] > self.slowd[-1] and
            self.data.Low[-1] < self.data.Low[-2] and
            self.obv[-1] > self.obv[-2] and
            current_close > self.swing_low[-1]):
            
            sl = current_close - (self.atr_multiplier_sl * atr)
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / (current_close - sl)))
            
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=sl,
                        tp=current_close + (self.atr_multiplier_tp * atr))
                print(f"🌙✨🚀 BULLISH CONFLUENCE! LONG {position_size} units @ {current_close:.2f}")

        # Moon Dev Short Entry Logic 🌙📉
        elif ((self.slowk[-2] >= 80 and self.slowk[-1] < 80) and  # Replaced crossbelow
              self.slowk[-1] < self.slowd[-1] and
              self.data.High[-1] > self.data.High[-2] and
              self.obv[-1] < self.obv[-2] and
              current_close < self.swing_high[-1]):
            
            sl = current_close + (self.atr_multiplier_sl * atr)
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / (sl - current_close)))
            
            if position_size > 0:
                self.sell(size=position_size,
                         sl=sl,
                         tp=current_close - (self.atr_multiplier_t