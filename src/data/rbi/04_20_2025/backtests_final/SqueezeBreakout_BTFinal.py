```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and prepare data
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
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class SqueezeBreakout(Strategy):
    def init(self):
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close)
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        # Squeeze calculations
        self.bb_width = self.I(lambda: (self.bb_upper - self.bb_lower) / self.bb_middle)
        self.bb_width_high = self.I(talib.MAX, self.bb_width, 20)
        self.bb_width_low = self.I(talib.MIN, self.bb_width, 20)
        
        # VWAP and Keltner
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume, length=12)
        self.ema_20 = self.I(talib.EMA, self.data.Close, 20)
        self.atr_20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.keltner_upper = self.I(lambda ema, atr: ema + 1.5 * atr, self.ema_20, self.atr_20)
        self.keltner_lower = self.I(lambda ema, atr: ema - 1.5 * atr, self.ema_20, self.atr_20)
        
        # Volume and swing levels
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        
        print("🌙 Moon Dev Strategy Activated! Ready for stellar trades! 🚀")

    def next(self):
        if len(self.data) < 20:
            return
        
        # Current values
        close = self.data.Close[-1]
        squeeze = self.bb_width[-1] < 0.5 * (self.bb_width_high[-1] - self.bb_width_low[-1])
        
        # Entry logic
        if not self.position:
            if squeeze:
                # Long entry
                if (self.data.Close[-2] < self.vwap[-2] and self.data.Close[-1] > self.vwap[-1]):
                    sl = self.swing_low[-1]
                    risk = 0.01 * self.equity
                    size = int(round(risk / (close - sl)))
                    if size > 0:
                        self.buy(size=size, sl=sl)
                        print(f"🚀🌙 LUNAR LIFT-OFF! Long {size} @ {close}, SL: {sl}")
                
                # Short entry
                elif (self.vwap[-2] < self.data.Close[-2] and self.vwap[-1] > self.data.Close[-1]):
                    sl = self.swing_high[-1]
                    risk = 0.01 * self.equity
                    size = int(round(risk / (sl - close)))
                    if size >