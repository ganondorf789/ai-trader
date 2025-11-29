Here's the fixed code with all the necessary adjustments and Moon Dev themed enhancements:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Data preparation 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
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

class VolCompressSentinel(Strategy):
    risk_pct = 0.01
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    lookback = 30
    vol_ma_period = 20
    
    def init(self):
        # Check required volatility data 🌩️
        if 'vix' not in self.data.df.columns or 'vvix' not in self.data.df.columns:
            raise ValueError("🌑 CRITICAL: VIX/VVIX data missing! Moon Dev systems require proper volatility feeds")

        # Volatility metrics 🌊
        self.vix = self.data.df['vix'].values
        self.vvix = self.data.df['vvix'].values
        self.vix_vvix_ratio = self.I(lambda x: x, self.vix/self.vvix, name='VIX/VVIX')
        self.ratio_low = self.I(talib.MIN, self.vix_vvix_ratio, timeperiod=self.lookback, name='Ratio 30L')

        # Bollinger Squeeze 📦
        self.upper, self.mid, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                 timeperiod=self.bb_period, 
                                                 nbdevup=self.bb_dev, 
                                                 nbdevdn=self.bb_dev,
                                                 matype=0)
        self.bb_width = self.I(lambda x: x, (self.upper-self.lower)/self.mid, name='BB Width')
        self.bb_low = self.I(talib.MIN, self.bb_width, timeperiod=self.lookback, name='BB Width 30L')

        # Volume confirmation 📉
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period, name='Vol MA')

        # Risk tools ⚖️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period, name='ATR')
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                         acceleration=0.02, maximum=0.2, name='SAR')

        print("🌙 Moon Dev Systems Armed - Volatility Compression Detector Active! 🚀")
        print("✨ All indicators initialized with pure TA-Lib power - No backtesting.lib contamination detected")

    def next(self):
        if len(self.data) < max(self.lookback, self.bb_period, self.atr_period):
            return  # Wait for sufficient data

        if self.position:
            self.manage_position()
        else:
            self.check_entries()

    def check_entries(self):
        # Core conditions check 🔍
        cond1 = self.vix_vvix_ratio[-1] <= self.ratio_low[-1]
        cond2 = self.bb_width[-1] <= self.bb_low[-1] 
        cond3 = self.data.Volume[-1] < self.vol_ma[-1]
        cond4 = self.vix[-1] > 12  # Volatility filter

        if all([cond1, cond2, cond3, cond4]):
            close = self.data.Close[-1]
            if close > self.upper[-1]:
                self.open_long()
            elif close < self.lower[-1]:
                self.open_short()

    def open_long(self):
        atr = self.atr[-1]
        entry = self.data.Close[-1]
        sl = entry - 0.5*atr
        tp = entry + 1.5*atr
        
        risk_amount = self.risk_pct * self.equity
        size = int(round(risk_amount / (entry - sl