```python
import pandas as pd
import talib as ta
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Calculate VWAP with daily anchor
data['datetime'] = pd.to_datetime(data['datetime'])
data['date'] = data['datetime'].dt.date
typical_price = (data['High'] + data['Low'] + data['Close']) / 3
data['cum_tpv'] = typical_price * data['Volume']
data['cum_volume'] = data['Volume']
data['cum_tpv'] = data.groupby('date')['cum_tpv'].cumsum()
data['cum_volume'] = data.groupby('date')['cum_volume'].cumsum()
data['vwap'] = data['cum_tpv'] / data['cum_volume']
data = data.drop(columns=['datetime', 'date', 'cum_tpv', 'cum_volume'])

class VoltaicSqueeze(Strategy):
    def init(self):
        # Bollinger Bands components
        self.sma20 = self.I(ta.SMA, self.data.Close, timeperiod=20)
        self.std20 = self.I(ta.STDDEV, self.data.Close, timeperiod=20)
        self.bb_upper = self.I(lambda sma, std: sma + 2*std, self.sma20, self.std20)
        self.bb_lower = self.I(lambda sma, std: sma - 2*std, self.sma20, self.std20)
        
        # Strategy indicators
        self.bb_width = self.I(lambda u,l: u-l, self.bb_upper, self.bb_lower)
        self.bb_width_sma = self.I(ta.SMA, self.bb_width, timeperiod=20)
        self.volume_sma = self.I(ta.SMA, self.data.Volume, timeperiod=10)
        self.entry_bar = 0

    def next(self):
        if not self.position:
            if self.bb_width[-1] < self.bb_width_sma[-1]:
                # Long entry logic
                if self.data.Close[-1] > self.bb_upper[-1]:
                    if (self.data.Volume[-1] >= 1.5*self.volume_sma[-1]) and (self.data.Close[-1] > self.data.vwap[-1]):
                        risk = 0.01 * self.equity
                        entry = self.data.Close[-1]
                        sl = self.bb_lower[-1]
                        risk_amount = entry - sl
                        if risk_amount <= 0: return
                        size = int(round(risk / risk_amount))
                        if size > 0:
                            self.buy(size=size, sl=sl, tp=entry + 2*(entry-sl))
                            self.entry_bar = len(self.data)
                            print(f"🌙✨🚀 MOON DEV LONG! Size: {size} | Entry: {entry:.2f} | SL: {sl:.2f} | TP: {entry + 2*(entry-sl):.2f}")
                
                # Short entry logic
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    if (self.data.Volume[-1] >= 1.5*self.volume_sma[-1]) and (self.data.Close[-1] < self.data.vwap[-1]):
                        risk = 0.01 * self.equity
                        entry = self.data.Close[-1]
                        sl = self.bb_upper[-1]
                        risk_amount = sl - entry
                        if risk_amount <= 0: return
                        size = int(round(risk / risk_amount))
                        if size > 0:
                            self.sell(size=size, sl=sl, tp=entry - 2*(sl-entry))
                            self.entry_bar = len(self.data)
                            print(f"🌙✨🚀 MOON DEV SHORT! Size: {size} | Entry: {entry:.2f} | SL: {sl:.2f} | TP: {entry - 2*(sl-entry):.2f}")
        else:
            # Time-based exit
            if len(self.data) - self