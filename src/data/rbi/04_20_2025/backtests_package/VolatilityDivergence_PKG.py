Here's the fixed code with proper indicator packages and Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilityDivergence(Strategy):
    ema_period = 20
    atr_period = 20
    keltner_mult = 2
    swing_period = 5
    risk_pct = 0.01
    atr_exit_threshold = 0.5
    sl_mult = 1.5

    def init(self):
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.upper_band = self.I(lambda e, a: e + self.keltner_mult * a, self.ema, self.atr)
        self.lower_band = self.I(lambda e, a: e - self.keltner_mult * a, self.ema, self.atr)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.price_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.price_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.obv_high = self.I(talib.MAX, self.obv, self.swing_period)
        self.obv_low = self.I(talib.MIN, self.obv, self.swing_period)
        self.atr_median = self.I(ta.median, self.atr, self.atr_period)
        self.exit_threshold = self.I(lambda x: x * self.atr_exit_threshold, self.atr_median)

    def next(self):
        if len(self.data) < max(self.swing_period, self.atr_period) + 2:
            return

        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        exit_threshold = self.exit_threshold[-1]

        if not self.position:
            # Long logic
            if (current_close > self.upper_band[-1] and
                self.price_low[-1] < self.price_low[-2] and
                self.obv_low[-1] > self.obv_low[-2]):
                
                sl_price = current_close - self.sl_mult * current_atr
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (current_close - sl_price)))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🌙🚀 MOON DEV LONG SIGNAL! Size: {position_size} @ {current_close:.2f} SL: {sl_price:.2f}")

            # Short logic
            elif (current_close < self.lower_band[-1] and
                  self.price_high[-1] > self.price_high[-2] and
                  self.obv_high[-1] < self.obv_high[-2]):
                
                sl_price = current_close + self.sl_mult * current_atr
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (sl_price - current_close)))
                if position_size > 0:
                    self.sell(size=position_size, sl=sl_price)
                    print(f"🌙🌒 MOON DEV SHORT SIGNAL! Size: {position_size} @ {current_close:.2f} SL: {sl_price:.2f}")

        elif current_atr < exit_threshold:
            self.position.close()
            print(f"🌙💤 MOON DEV VOLATILITY EXIT! Closed @ {current_close:.2f} (ATR: {current_atr:.2f})")

bt