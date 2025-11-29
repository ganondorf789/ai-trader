```python
# 🌙 Moon Dev's VoltaicFibreversal Backtest Implementation
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Clean and prepare data 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VoltaicFibreversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    atr_period = 14
    swing_window = 20
    divergence_window = 5
    rsi_period = 14

    def init(self):
        # Core indicators ✨
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Divergence detection tools 🔍
        self.price_mins = self.I(talib.MIN, self.data.Low, self.divergence_window)
        self.rsi_mins = self.I(talib.MIN, self.rsi, self.divergence_window)

        # Volatility system 🌪️
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_period)
        
        self.fib_targets = []

    def next(self):
        if self.position:
            self.manage_position()
        else:
            self.check_entries()

    def check_entries(self):
        # Bullish divergence check 📈
        if len(self.price_mins) < 2 or len(self.rsi_mins) < 2:
            return

        price_divergence = (self.price_mins[-1] < self.price_mins[-2]) 
        rsi_divergence = (self.rsi_mins[-1] > self.rsi_mins[-2])
        
        # IV breach confirmation 🚀
        iv_breach = self.data.Close[-1] > self.swing_high[-1]

        if price_divergence and rsi_divergence and iv_breach:
            self.enter_trade()

    def enter_trade(self):
        # Risk calculations 🔒
        entry_price = self.data.Close[-1]
        stop_price = self.price_mins[-1] * 0.995  # 0.5% below divergence low
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = entry_price - stop_price
        
        if risk_per_unit <= 0:
            return  # Avoid invalid calculation 🌙

        position_size = int(round(risk_amount / risk_per_unit))
        
        # Fib targets calculation 🎯
        swing_range = self.swing_high[-1] - self.price_mins[-1]
        self.fib_targets = [
            entry_price + swing_range * 0.382,
            entry_price + swing_range * 0.5,
            entry_price + swing_range * 0.618
        ]

        # Execute entry 🌙
        self.buy(
            size=position_size,
            sl=stop_price,
            tag=f"🌕 VOLTAIC ENTRY | Size: {position_size} | Risk: {self.risk_pct*100}%"
        )
        
        print(f"🚀 MOON DEV ALERT: Long Entry @ {entry_price:.2f}")
        print(f"🎯 Fib Targets: 38.2% {self.fib_targets[0]:.2f} | 50% {self.fib_targets[1]:.2f} | 61.8% {self.fib_targets[2]:.2f}")

    def manage_position(self):
        price = self.data.Close[-1]
        for target in self.fib_targets:
            if price >= target:
                self.position.close()
                print(f"🌙 MOON DEV ALERT: Target Hit @ {