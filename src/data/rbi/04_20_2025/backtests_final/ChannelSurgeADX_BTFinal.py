```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class ChannelSurgeADX(Strategy):
    risk_per_trade = 0.01
    consec_losses = 0
    max_consec_losses = 3
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.donchian_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.donchian_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.di_plus = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.di_minus = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def next(self):
        if len(self.data) < 20 or self.consec_losses >= self.max_consec_losses:
            return

        current_close = self.data.Close[-1]
        adx_rising = self.adx[-1] > self.adx[-2]
        
        # Long entry conditions
        if (current_close > self.donchian_high[-2] and 
            self.adx[-1] > 25 and 
            adx_rising and 
            self.di_plus[-1] > self.di_minus[-1]):
            
            if not self.position:
                risk_amount = self.equity * self.risk_per_trade
                atr_value = self.atr[-1]
                stop_loss = current_close - atr_value
                take_profit = current_close + 2 * atr_value
                
                position_size = int(round(risk_amount / (current_close - stop_loss)))
                if position_size > 0:
                    print(f"🌙 MOON DEV LONG ENTRY! 🚀 Price: {current_close:.2f}, Size: {position_size} units")
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
        
        # Short entry conditions
        elif (current_close < self.donchian_low[-2] and 
              self.adx[-1] > 25 and 
              adx_rising and 
              self.di_minus[-1] > self.di_plus[-1]):
            
            if not self.position:
                risk_amount = self.equity * self.risk_per_trade
                atr_value = self.atr[-1]
                stop_loss = current_close + atr_value
                take_profit = current_close - 2 * atr_value
                
                position_size = int(round(risk_amount / (stop_loss - current_close)))
                if position_size > 0:
                    print(f"🌙 MOON DEV SHORT ENTRY! 🌑 Price: {current_close:.2f}, Size: {position_size} units")
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
        
        # Check for secondary exits
        if self.position:
            if (self.adx[-1] < 25 or 
                (self.position.is_long and current_close < self.donchian_high[-1]) or 
                (self.position.is_short and current_close > self.donchian_low[-1])):
                
                print(f"✨ MOON DEV SECONDARY EXIT! 🔄 Closing {self.position.type}")
                self.position.close()
                
    def notify_trade(self, trade):
        if trade.is_closed:
            if trade.pnl < 0:
                self.consec_losses += 1
                print(f"💸 MOON DEV LOSS DETECTED! Consecutive losses: {self.consec_losses}/3")
            else:
                self.consec_losses = 0
                print(f"💰 MOON DEV PROFIT! ✨ Reset loss counter")

# Data handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'date': 'Date', 'open