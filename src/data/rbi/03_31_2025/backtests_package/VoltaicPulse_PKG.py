Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VoltaicPulse Backtest Implementation
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VoltaicPulse(Strategy):
    # 🌌 Strategy Parameters
    atr_period = 14
    atr_sma_period = 20
    vix_lookback = 960  # 10 days in 15m intervals (10*24*4=960)
    risk_percent = 0.01
    trailing_multiplier = 2

    def init(self):
        # 🌀 Indicator Calculations
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)
        self.vix_low = self.I(talib.MIN, self.data['vix'], timeperiod=self.vix_lookback)
        
        # 🛸 Track highest high during trade
        self.max_high = None

    def next(self):
        current_price = self.data.Close[-1]
        
        # 🌠 Moon Dev Debug Prints
        print(f"\n🌙 Moon Dev Pulse Check [{self.data.index[-1]}]")
        print(f"✨ ATR({self.atr_period}): {self.atr[-1]:.2f} | ATR SMA: {self.atr_sma[-1]:.2f}")
        print(f"🌪️ VIX: {self.data['vix'][-1]:.2f} | 10D Low: {self.vix_low[-1]:.2f}")

        # 🚀 Entry Logic
        if not self.position:
            if (self.atr[-2] < self.atr_sma[-2] and self.atr[-1] > self.atr_sma[-1]) and (self.data['vix'][-1] < self.vix_low[-1]):
                # 💰 Risk Management Calculation
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                risk_per_unit = self.trailing_multiplier * atr_value
                position_size = int(round(risk_amount / risk_per_unit))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.max_high = self.data.High[-1]
                    print(f"\n🚀⚡ VOLTAIC SURGE ACTIVATED! ⚡🚀")
                    print(f"📈 Entry Price: {current_price:.2f}")
                    print(f"🔢 Size: {position_size} units | Risk: {self.risk_percent*100}%")
                    print(f"🛡️ Initial ATR Stop: {atr_value*2:.2f}")

        # 🔥 Exit Logic
        else:
            self.max_high = max(self.max_high, self.data.High[-1])
            current_atr = self.atr[-1]
            stop_price = self.max_high - (current_atr * self.trailing_multiplier)
            
            print(f"🌕 Trail Stop Update | High Since Entry: {self.max_high:.2f}")
            print(f"🔻 Current Stop: {stop_price:.2f} ({current_atr*2:.2f} below peak)")
            
            if self.data.Low[-1] < stop_price:
                self.position.close()
                print(f"\n🌑 VOLTAIC DRAIN TRIGGERED! Closing position")
                print(f"🏁 Exit Price: {current_price:.2f} | PnL: {self.position.pl:.2f}")

# 🛰️ Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# � Data Cleaning
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# 🚨 Validate VIX Data Presence
if 'vix' not in data.columns:
    raise