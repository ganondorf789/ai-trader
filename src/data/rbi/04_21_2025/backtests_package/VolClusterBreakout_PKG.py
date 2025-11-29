```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data.index = pd.to_datetime(data['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

class VolClusterBreakout(Strategy):
    risk_percent = 0.01
    swing_period = 20
    
    def init(self):
        # Volatility indicators
        close = self.data.Close
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BBANDS')
        bbw = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bbw = self.I(lambda: bbw, name='BBW')
        self.bbw_percentile = self.I(lambda: bbw.rolling(200).apply(lambda x: np.percentile(x, 10)), name='BBW_PCT')
        
        # Volume indicator
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')
        
        # MACD
        self.macd, self.macd_signal, _ = self.I(talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9, name='MACD')
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='Swing_Low')
        
    def next(self):
        if len(self.data) < 200:
            return
        
        # Current values
        price = self.data.Close[-1]
        bbw = self.bbw[-1]
        bbw_pct = self.bbw_percentile[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        macd = self.macd[-1]
        macd_signal = self.macd_signal[-1]
        
        # Entry conditions
        long_cond = (
            (bbw > bbw_pct) and
            (self.bbw[-2] <= bbw_pct) and
            (volume > 1.5 * volume_ma) and
            (macd > macd_signal) and
            (price > self.swing_high[-1])
        )
        
        short_cond = (
            (bbw > bbw_pct) and
            (self.bbw[-2] <= bbw_pct) and
            (volume > 1.5 * volume_ma) and
            (macd < macd_signal) and
            (price < self.swing_low[-1])
        )
        
        # Risk management
        if long_cond and not self.position:
            sl = self.swing_low[-1]
            risk = price - sl
            if risk <= 0:
                return
            risk_amount = self.equity * self.risk_percent
            size = int(round(risk_amount / risk))
            if size > 0:
                self.buy(size=size, sl=sl, tp=price + 2*risk)
                print(f"🌙✨ MOON DEV LONG SIGNAL! Buying {size} @ {price:.2f} | SL: {sl:.2f} 🚀")
        
        elif short_cond and not self.position:
            sl = self.swing_high[-1]
            risk = sl - price
            if risk <= 0:
                return
            risk_amount = self.equity * self.risk_percent
            size = int(round(risk_amount / risk))
            if size > 0:
                self.sell(size=size, sl=sl, tp=price - 2*risk)
                print(f"🌙✨ MOON