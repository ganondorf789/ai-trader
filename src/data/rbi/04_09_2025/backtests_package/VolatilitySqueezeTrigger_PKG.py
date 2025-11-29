Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeTrigger(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands
        def bb_upper(close):
            return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[0]
        self.bb_upper = self.I(bb_upper, self.data.Close)
        
        def bb_lower(close):
            return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[2]
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, 20)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 50)
        
        # MACD
        def macd_line(close):
            return talib.MACD(close, 12, 26, 9)[0]
        self.macd = self.I(macd_line, self.data.Close)
        
        def macd_signal(close):
            return talib.MACD(close, 12, 26, 9)[1]
        self.signal = self.I(macd_signal, self.data.Close)
        
        # Keltner Channels
        self.keltner_mid = self.I(talib.EMA, self.data.Close, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.keltner_upper = self.I(lambda ema, atr: ema + 2*atr, self.keltner_mid, self.atr)
        self.keltner_lower = self.I(lambda ema, atr: ema - 2*atr, self.keltner_mid, self.atr)
        
        # Trade tracking
        self.trades_today = 0
        self.current_date = None

    def next(self):
        # Reset daily trade counter
        current_date = self.data.index[-1].date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.trades_today = 0
            print(f"\n🌙 New Moon Cycle: {current_date.strftime('%Y-%m-%d')} | Trades reset to 0")

        if self.trades_today >= 3:
            return

        # Entry logic
        bb_squeeze = self.bb_width[-1] < self.bb_width_sma[-1]
        volume_spike = self.data.Volume[-1] > self.volume_ma[-1]
        macd_bull = (self.macd[-2] < self.signal[-2]) and (self.macd[-1] > self.signal[-1])  # Replaced crossover
        macd_bear = (self.signal[-2] < self.macd[-2]) and (self.signal[-1] > self.macd[-1])  # Replaced crossover

        # Long entry
        if not self.position and bb_squeeze and volume_spike and macd_bull:
            bbw = self.bb_width[-1]
            sl_price = self.data.Close[-1] - 1.5 * bbw
            risk = self.data.Close[-1] - sl_price
            size = int(round((self.equity * self.risk_per_trade) / risk))
            
            if size > 0:
                self.buy(size=size, tag="🌕 BULLISH SQUEEZE