```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

# 🌙 Moon Dev Data Preparation Magic
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🧹 Cleanse columns with Moon Dev precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexVolume(Strategy):
    atr_multiplier = 3
    risk_pct = 0.01  # 1% risk per trade 🌙
    vi_period = 14
    vol_ma_period = 20

    def init(self):
        # ✨ Moon Dev Indicator Setup
        h, l, c = self.data.High, self.data.Low, self.data.Close
        
        # Vortex Indicator Calculation
        vortex = ta.vortex(h, l, c, length=self.vi_period)
        self.vi_plus = self.I(lambda: vortex[f'VTX_{self.vi_period}+'], name='VI+')
        self.vi_minus = self.I(lambda: vortex[f'VTX_{self.vi_period}-'], name='VI-')
        
        # Volume Confirmation
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period, name='Vol MA')
        
        # 🛡️ Chandelier Exit Components
        self.atr = self.I(talib.ATR, h, l, c, 14, name='ATR')

    def next(self):
        # 🌙 Avoid trading during system warmup
        if len(self.data) < max(self.vi_period, self.vol_ma_period) + 1:
            return

        # 🚀 Entry Logic with Moon Dev Flair
        if not self.position:
            # Bullish Entry
            if crossover(self.vi_plus, self.vi_minus) and self.data.Volume[-1] < self.vol_ma[-1]:
                risk_amount = self.equity * self.risk_pct
                atr = self.atr[-1] or 1  # Prevent zero division
                size = int(round(risk_amount / (atr * self.atr_multiplier)))
                if size:
                    self.buy(size=size)
                    print(f"🚀 MOON DEV LONG: {size} units @ {self.data.Close[-1]:.2f} | VI+ {self.vi_plus[-1]:.2f} > VI- {self.vi_minus[-1]:.2f} 🌕")
            
            # Bearish Entry
            elif crossover(self.vi_minus, self.vi_plus) and self.data.Volume[-1] < self.vol_ma[-1]:
                risk_amount = self.equity * self.risk_pct
                atr = self.atr[-1] or 1
                size = int(round(risk_amount / (atr * self.atr_multiplier)))
                if size:
                    self.sell(size=size)
                    print(f"🌑 MOON DEV SHORT: {size} units @ {self.data.Close[-1]:.2f} | VI- {self.vi_minus[-1]:.2f} > VI+ {self.vi_plus[-1]:.2f} 🌘")

        # 🛑 Chandelier Exit Logic
        for trade in self.trades:
            if trade.is_long:
                entry_idx = self.data.index.get_loc(trade.entry_time)
                high_since = self.data.High[entry_idx:].max()
                stop = high_since - self.atr[-1] * self.atr_multiplier
                if self.data.Low[-1] < stop:
                    trade.close()
                    print(f"🛑 MOON DEV LONG EXIT: Trailing stop @ {stop:.2f} | PNL: {trade.pl_pct:.2f}% ✨")
            
            elif trade.is_short:
                entry_idx = self.data.index.get_loc(trade.entry_time)
                low_since = self.data.Low[entry_idx:].min()
                stop = low_since + self.atr[-