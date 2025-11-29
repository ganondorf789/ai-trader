```python
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index('datetime')

class KeltnerDivergence(Strategy):
    risk_per_trade = 0.01
    max_positions = 3
    daily_loss_limit = -0.03
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.upper_band = self.I(lambda: self.ema + 1.5 * self.atr)
        self.lower_band = self.I(lambda: self.ema - 1.5 * self.atr)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # Track daily metrics
        self.current_day = None
        self.daily_pnl = 0
        self.daily_limit_reached = False
        
    def next(self):
        # Moon Dev's Daily Loss Check 🌙
        current_date = self.data.index[-1].date()
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_pnl = 0
            self.daily_limit_reached = False
            
        if self.daily_limit_reached:
            return
        
        # Moon Dev's Position Management 🌙
        if len(self.trades) >= self.max_positions:
            return
        
        # Moon Dev's Signal Check 🌙
        if len(self.data.Close) < 3:
            return
        
        # Bullish Divergence Check 🌙✨
        prev_low = self.data.Low[-2]
        prev_prev_low = self.data.Low[-3]
        rsi_val = self.rsi[-2]
        rsi_prev = self.rsi[-3]
        
        bullish_divergence = (prev_low < prev_prev_low) and (rsi_val > rsi_prev)
        bearish_divergence = (self.data.High[-2] > self.data.High[-3]) and (self.rsi[-2] < self.rsi[-3])
        
        # Moon Dev's Entry Logic 🌙🚀
        if (self.data.Close[-2] < self.lower_band[-2] and 
            bullish_divergence and 
            self.rsi[-2] < 40):
            
            # Risk Management Calculation 🌙⚖️
            entry_price = self.data.Open[-1]
            atr_value = self.atr[-1]
            stop_loss = entry_price - atr_value
            take_profit = entry_price + 2 * atr_value
            
            position_size = int(round((self.risk_per_trade * self.equity) / 
                                (entry_price - stop_loss)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                print(f"🌙🚀 BULLISH ENTRY @ {entry_price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        
        # Bearish Entry Logic 🌙🛑
        elif (self.data.Close[-2] > self.upper_band[-2] and 
              bearish_divergence and 
              self.rsi[-2] > 60):
            
            entry_price = self.data.Open[-1]
            atr_value = self.atr[-1]
            stop_loss = entry_price + atr_value
            take_profit = entry_price - 2 * atr_value
            
            position_size = int(round((self.risk_per_trade * self.equity) / 
                                (stop_loss - entry_price)))
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                print(f"🌙