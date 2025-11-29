```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import os

class VoltaicSqueeze(Strategy):
    def init(self):
        # Calculate all indicators using self.I() wrapper
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Calculate Bollinger Band width
        self.bb_width = self.I(lambda u, m, l: (u - l)/m, 
            self.bb_upper, self.bb_middle, self.bb_lower)
        
        # 30-period minimum of BB width
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=30)
        
        # Momentum indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, 
            self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # BB width moving average for exit condition
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=30)
        
        self.take_profit = None
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        # Moon Dev progress tracker
        if len(self.data) % 1000 == 0:
            print(f"🌓 Processing Bar {len(self.data)} | "
                  f"Equity: ${self.equity:,.2f}")

        # Entry Logic
        if not self.position:
            current_bb_width = self.bb_width[-1]
            current_min = self.bb_width_min[-1]
            current_rsi = self.rsi[-1]

            # Check squeeze condition and momentum
            if abs(current_bb_width - current_min) < 1e-5 and current_rsi > 50:
                close_price = self.data.Close[-1]
                atr_value = self.atr[-1]
                
                # Moon Dev Risk Management
                risk_percent = 1.0  # Using full equity
                position_size = (self.equity * risk_percent) / close_price
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.take_profit = close_price + 2 * atr_value
                    self.buy(size=position_size)
                    print(f"🚀 MOON DEV LONG SIGNAL 🌙 | "
                          f"Size: {position_size} | "
                          f"TP: ${self.take_profit:.2f}")

        # Exit Logic        
        else:
            # Take profit check
            if self.data.High[-1] >= self.take_profit:
                self.position.close()
                print(f"🌕 PROFIT TARGET REACHED 🎯 | "
                      f"Price: ${self.data.Close[-1]:.2f}")
            
            # Bollinger width expansion exit
            elif self.bb_width[-1] > self.bb_width_avg[-1]:
                self.position.close()
                print(f"🌑 BB WIDTH EXPANSION EXIT 📉 | "
                      f"Width: {self.bb_width[-1]:.2f}")

# Data preparation
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Execute backtest
bt = Backtest(data, VoltaicSqueeze, 
    cash=1_000_000, commission=.002, exclusive_orders=True)
stats = bt.run()

# Moon Dev Final Report
print("\n🌕🌖🌗🌘🌑🌒🌓🌔🌕")
print("MOON DEV BACKTEST COMPLETE 🚀")
print("="*40)
print(stats)
print("\nSTRATEGY ANALYSIS:")
print(stats._strategy)
print("="*40)
print("MOON DEV TRADING WISDOM: "