```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
from backtesting.test import EURUSD

# Moon Dev Data Preparation 🌙
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Ensure proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

class VolClusterReversion(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    high_period = 20
    low_period = 10
    atr_multiplier_entry = 2
    atr_multiplier_exit = 1
    
    def init(self):
        # Moon Dev Indicator Setup 🌙✨
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.high_20 = self.I(talib.MAX, self.data.High, timeperiod=self.high_period)
        self.low_10 = self.I(talib.MIN, self.data.Low, timeperiod=self.low_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20)  # Volatility filter
        
        print("🌙 MOON DEV STRATEGY INITIALIZED 🌙")
        print(f"| ATR Period: {self.atr_period}")
        print(f"| High Period: {self.high_period}")
        print(f"| Low Period: {self.low_period}")
        print("✨ READY FOR LIFTOFF! 🚀\n")

    def next(self):
        price = self.data.Close[-1]
        atr_value = self.atr[-1]
        high_20_value = self.high_20[-1]
        low_10_value = self.low_10[-1]
        atr_ma_value = self.atr_ma[-1]

        # Moon Dev Volatility Check 🌙⚡
        if atr_value < atr_ma_value:
            print("🌑 VOLATILITY TOO LOW - STANDING ASIDE")
            return

        # Calculate trading signals
        long_entry_level = high_20_value - self.atr_multiplier_entry * atr_value
        exit_level = low_10_value + self.atr_multiplier_exit * atr_value

        # Moon Dev Position Management 🌙💼
        if not self.position:
            if price <= long_entry_level:
                # Risk Management Calculations
                stop_loss = price - 2 * atr_value
                risk_amount = self.risk_per_trade * self.equity
                risk_per_unit = price - stop_loss
                position_size = int(round(risk_amount / risk_per_unit))
                
                if position_size > 0:
                    print(f"\n🚀 MOON DEV LONG SIGNAL 🌙")
                    print(f"| Price: {price:.2f}")
                    print(f"| Entry Level: {long_entry_level:.2f}")
                    print(f"| Position Size: {position_size}")
                    print(f"| Stop Loss: {stop_loss:.2f}")
                    print(f"| Take Profit: {exit_level:.2f}")
                    self.buy(size=position_size, sl=stop_loss, tp=exit_level)
        
        else:
            if price >= exit_level:
                print(f"\n🌕 MOON DEV EXIT SIGNAL 🌙")
                print(f"| Price reached target: {price:.2f}")
                self.position.close()

# Moon Dev Backtest Execution 🌙🚀
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = load_data(data_path)

print("🌙 INITIATING MOON DEV BACKTEST 🌙")
print("|----------------------------------")
print(f"| Data Points: {len(data)}")
print(f"| Start Date: {data.index[0]}")
print(f"| End Date: {data.index[-1]}")
print("|----------------------------------")
print("✨ I