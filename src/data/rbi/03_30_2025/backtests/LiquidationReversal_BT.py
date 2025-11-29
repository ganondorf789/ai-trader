```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Clean and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class LiquidationReversal(Strategy):
    def init(self):
        # Core Indicators 🌙
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume MA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(talib.BBANDS, 
            self.data.Close, 20, 2, 2, name=['Upper BB', 'Middle BB', 'Lower BB'])
        self.engulfing = self.I(talib.CDLENGULFING, 
            self.data.Open, self.data.High, self.data.Low, self.data.Close, name='Engulfing')
        
        print("🌙 Lunar Strategy Activated! Ready to hunt liquidations! 🚀")

    def next(self):
        current_idx = len(self.data) - 1
        
        # Manage existing positions ✨
        if self.position:
            if self.position.is_long and self.data.Close[-1] > self.upper_bb[-1]:
                self.position.close()
                print(f"🌕 LONG Closed at {self.data.Close[-1]} - Upper BB Touch!")
            elif self.position.is_short and self.data.Close[-1] < self.lower_bb[-1]:
                self.position.close()
                print(f"🌑 SHORT Closed at {self.data.Close[-1]} - Lower BB Touch!")
            return

        # Entry Logic - Short Signals 🌪️
        if (self.data.High[-1] == self.swing_high[-1] and
            self.data.Volume[-1] > 2*self.volume_ma[-1] and
            self.engulfing[-1] == -100):
            
            recent_volatility = (self.data.High[-3:].max() - self.data.Low[-3:].min())/self.data.Low[-3:].min()
            if recent_volatility >= 0.02:
                sl = self.swing_high[-1] + 2*self.atr[-1]
                risk_amount = 0.01 * self.equity
                position_size = int(round(risk_amount/(sl - self.data.Close[-1])))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl, 
                            tp=self.data.Close[-1] - 2*(sl - self.data.Close[-1]))
                    print(f"🌑🚀 SHORT ENTRY! Size: {position_size} | SL: {sl:.2f} | Reward/Risk: 2:1")

        # Entry Logic - Long Signals 🌟
        elif (self.data.Low[-1] == self.swing_low[-1] and
              self.data.Volume[-1] > 2*self.volume_ma[-1] and
              self.engulfing[-1] == 100):
            
            recent_volatility = (self.data.High[-3:].max() - self.data.Low[-3:].min())/self.data.Low[-3:].min()
            if recent_volatility >= 0.02:
                sl = self.swing_low[-1] - 2*self.atr[-1]
                risk_amount = 0.01 * self.equity
                position_size = int(round(risk_amount/(self.data.Close[-1] - sl)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=