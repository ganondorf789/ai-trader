```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class ChaikinSurgeTrail(Strategy):
    def init(self):
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 21)
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.highest_close = self.lowest_close = self.entry_price = 0

    def next(self):
        if len(self.data.Close) < 20 or self.atr[-1] == 0:
            return

        current_cmf = self.cmf[-1]
        prev_cmf = self.cmf[-2] if len(self.cmf) > 1 else 0
        volume_surge = self.data.Volume[-1] > 1.5 * self.volume_avg[-1]
        atr_value = self.atr[-1]

        if not self.position:
            if self.atr[-1] > 0.03 * self.data.Close[-1]:
                print("🌙🚫 Volatility filter blocked trade: ATR too high!")
                return

            if current_cmf > 0 and prev_cmf <= 0 and volume_surge:
                risk_amount = self.equity * 0.01
                risk_per_share = 2 * atr_value
                position_size = int(round(risk_amount / risk_per_share)) if risk_per_share else 0
                if position_size:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.highest_close = self.entry_price
                    print(f"🌙🚀 LONG ENTRY! Size: {position_size} @ {self.entry_price:.2f}")

            elif current_cmf < 0 and prev_cmf >= 0 and volume_surge:
                risk_amount = self.equity * 0.01
                risk_per_share = 2 * atr_value
                position_size = int(round(risk_amount / risk_per_share)) if risk_per_share else 0
                if position_size:
                    self.sell(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.lowest_close = self.entry_price
                    print(f"🌙📉 SHORT ENTRY! Size: {position_size} @ {self.entry_price:.2f}")

        else:
            if self.position.is_long:
                self.highest_close = max(self.highest_close, self.data.Close[-1])
                trailing_stop = self.highest_close - 3 * atr_value
                current_stop = max(self.entry_price - 2 * atr_value, trailing_stop)

                if self.data.Low[-1] <= current_stop:
                    self.position.close()
                    print(f"🌙🛑 Trailing stop hit! Closed long @ {self.data.Close[-1]:.2f}")
                elif current_cmf < 0 and prev_cmf >= 0:
                    self.position.close()
                    print(f"🌙🔻 CMF reversal! Closed long @ {self.data.Close[-1]:.2f}")

            elif self.position.is_short:
                self.lowest_close = min(self.lowest_close, self.data.Close[-1])
                trailing_stop = self.lowest_close + 3 * atr_value
                current_stop = min(self.entry_price + 2 * atr_value, trailing_stop)

                if self.data.High[-1] >= current_stop:
                    self.position.close()
                    print(f"🌙🛑 Trailing stop hit! Closed short @ {self.data.Close[-1]:.2f}")
                elif current_cmf > 0 and prev_cmf <= 0:
                    self.position.close()
                    print(f"🌙🔺 CMF reversal! Closed short @ {self.data.Close[-1]:.2f}")

data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime