Here's the debugged and fixed version of your ChikouDivergence strategy with Moon Dev themed debug prints and proper technical fixes:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class ChikouDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using TA-Lib wrapped with self.I()
        self.chikou = self.I(lambda c: c.shift(26), self.data.Close, name='Chikou')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        self.macd_hist = self.I(lambda c: talib.MACD(c, 12, 26, 9)[2], self.data.Close, name='MACD Hist')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='Swing Low')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='Swing High')
        
    def next(self):
        # Wait until we have enough historical data
        if len(self.data) < 30:
            return
        
        current_close = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        
        if not self.position:
            # Bullish entry conditions 🌙
            bull_div = (current_low < self.data.Low[-2] and
                       self.chikou[-1] > self.chikou[-2])
            bull_conf = (current_close > self.data.High[-2] and
                         self.rsi[-1] < 30)
            
            if bull_div and bull_conf:
                sl_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_percent
                distance = current_close - sl_price
                if distance > 0:
                    size = int(round(risk_amount / distance))
                    if size > 0:
                        self.buy(size=size, sl=sl_price, 
                                tp=current_close + 2*distance)
                        print(f"🌙✨ BULLISH DIVERGENCE DETECTED! Long {size} @ {current_close} 🚀")
            
            # Bearish entry conditions 🌙
            bear_div = (current_high > self.data.High[-2] and
                        self.chikou[-1] < self.chikou[-2])
            bear_conf = (current_close < self.data.Low[-2] and
                         self.rsi[-1] > 70)
            
            if bear_div and bear_conf:
                sl_price = self.swing_high[-1]
                risk_amount = self.equity * self.risk_percent
                distance = sl_price - current_close
                if distance > 0:
                    size = int(round(risk_amount / distance))
                    if size > 0:
                        self.sell(size=size, sl=sl_price,
                                 tp=current_close - 2*distance)
                        print(f"🌙✨ BEARISH DIVERGENCE SPOTTED! Short {size} @ {current_close} 🌑")
        
        else:
            # Exit conditions with MACD histogram divergence 🌙
            if self.position.is_long:
                if (current_high > self.data.High[-2] and
                    self.macd_hist[-1] < self.macd_hist[-2]):
                    self.position.close()
                    print(f"🌙 MACD BEARISH DIVERGENCE! Closing long @ {current_close} 💰")
                    
            elif self.position.is_short:
                if (current_low < self.data.Low[-2] and
                    self.macd_hist[-1] > self.macd_hist[-2]):
                    self.position.close()
                    print(f"🌙 MACD BULLISH DIVERGENCE! Closing short @ {current_close} 💰")

# Data preparation 🌙
print("🌙 Preparing lunar data for analysis...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['