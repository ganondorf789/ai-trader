Here's the fully debugged version of your Vortex Compression strategy with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from datetime import time

# Load and preprocess data 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data ✨
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
data = data.set_index('datetime')

class VortexCompression(Strategy):
    risk_pct = 0.01
    vi_period = 14
    vol_sma_period = 20
    bb_period = 20
    bw_sma_period = 50
    atr_period = 14

    def init(self):
        # Vortex Indicator 🌪️
        vi = ta.vortex(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close, 
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi[f'VORTIC_{self.vi_period}_1'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VORTIC_{self.vi_period}_2'], name='VI-')

        # Volume Confirmation 🔊
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)

        # Bollinger Bandwidth Compression 🔧
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=self.bb_period)
        self.bandwidth = self.I(lambda: (upper - lower) / middle)
        self.bw_sma = self.I(talib.SMA, self.bandwidth, timeperiod=self.bw_sma_period)

        # Risk Management 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

    def next(self):
        if len(self.data) < max(self.vi_period, self.bw_sma_period, 50):
            return

        # Time filter ⏰
        current_time = self.data.index[-1].time()
        if (current_time >= time(0, 0) and current_time <= time(0, 30)) or \
           (current_time >= time(23, 30)):
            return

        # Get current values
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        bw = self.bandwidth[-1]
        bw_sma = self.bw_sma[-1]
        atr = self.atr[-1]

        # Entry Logic 🚪
        if not self.position:
            # Long Entry 🚀
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
                vol > vol_sma and
                bw < bw_sma):
                
                sl = price - 1.5 * atr
                risk_amount = self.risk_pct * self.equity
                position_size = int(round(risk_amount / (price - sl)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl)
                    print(f"🌙 MOON DEV LONG ENTRY ✨ | Price: {price:.2f} | Size: {position_size} | SL: {sl:.2f} | VI+ {self.vi_plus[-1]:.2f} > VI- {self.vi_minus[-1]:.2f}")

            # Short Entry 🌪️
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and
                  vol > vol_sma and
                  bw < bw_sma):
                
                sl = price + 1.5 * atr
                risk_amount = self.risk_pct * self.equity
                position_size = int(