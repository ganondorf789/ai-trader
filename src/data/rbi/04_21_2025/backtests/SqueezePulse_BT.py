```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

# Load and preprocess data
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

class SqueezePulse(Strategy):
    def init(self):
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_mid = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        
        # Volume-Weighted MACD
        vw_price = self.data.Close * self.data.Volume
        self.macd = self.I(talib.MACD, vw_price, fastperiod=12, slowperiod=50, signalperiod=9)[0]
        self.signal = self.I(talib.MACD, vw_price, fastperiod=12, slowperiod=50, signalperiod=9)[1]
        
        # Swing highs/lows for squeeze detection
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)

    def next(self):
        if len(self.data) < 50:
            return
        
        # Moon Dev debug prints 🌙
        print(f"🌙 Current Close: {self.data.Close[-1]} | Upper BB: {self.bb_upper[-1]:.2f} | Lower BB: {self.bb_lower[-1]:.2f}")
        
        # Squeeze detection
        squeeze = (self.swing_high[-1] - self.swing_low[-1]) < 0.5*(self.bb_upper[-1] - self.bb_lower[-1])
        
        # Entry conditions
        if not self.position:
            # Long entry 🌙🚀
            if (self.data.Close[-1] > self.bb_upper[-1] and 
                crossover(self.macd, self.signal) and 
                squeeze):
                risk_amount = self.equity * 0.01
                sl = self.bb_lower[-1]
                risk_per_share = self.data.Close[-1] - sl
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl)
                    print(f"🌙🚀 MOON ENTRY LONG | Size: {position_size} | SL: {sl:.2f}")
            
            # Short entry 🌙📉
            elif (self.data.Close[-1] < self.bb_lower[-1] and 
                  crossover(self.signal, self.macd) and 
                  squeeze):
                risk_amount = self.equity * 0.01
                sl = self.bb_upper[-1]
                risk_per_share = sl - self.data.Close[-1]
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.sell(size=position_size, sl=sl)
                    print(f"🌙📉 MOON ENTRY SHORT | Size: {position_size} | SL: {sl:.2f}")
        
        # Exit conditions 🌙✨
        else:
            if self.position.is_long:
                if self.data.Close[-1] < self.bb_mid[-1]:
                    self.position.close()
                    print(f"🌙✨ MOON EXIT LONG | Price: {self.data.Close[-1]:.2f}")
            elif self.position.is_short:
                if self.data.Close[-1] > self.bb_mid[-1]:
                    self.position.close()
                    print(f"🌙✨ MO