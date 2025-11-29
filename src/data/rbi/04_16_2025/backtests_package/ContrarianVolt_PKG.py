I've analyzed the code and found it's already clean from any `backtesting.lib` imports or functions! 🌟 The strategy correctly uses TA-Lib for indicators and proper array indexing for conditions. Here's the verified code with some Moon Dev enhancements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest

# Data preprocessing function with Moon Dev magic ✨
def prepare_data(path):
    print("🌙 MOON DEV DATA PREP: Loading cosmic market data...")
    data = pd.read_csv(path)
    # Clean column names with lunar precision
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns (cosmic debris)
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Rename columns to match backtesting requirements
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    print("🌌 Data preparation complete! Ready for lunar strategy deployment.")
    return data

# Load and prepare data with celestial path
DATA_PATH = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = prepare_data(DATA_PATH)

class ContrarianVolt(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade (cosmic risk management)
    
    def init(self):
        print("🌕 MOON DEV INIT: Charging lunar indicators...")
        # Calculate indicators using TA-Lib with self.I()
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14)
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.upper_band = self.I(self._calc_upper_band, self.data.Close)
        print("✨ Indicator initialization complete! Ready for cosmic signals.")
        
    def _calc_upper_band(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper
        
    def next(self):
        if not self.position:
            # Entry conditions with lunar precision
            cmo_condition = (self.cmo[-1] < -50) and (self.cmo[-2] >= -50)  # No crossover needed!
            funding_condition = (self.data.funding_rate[-1] < -0.001) and \
                               (self.data.funding_rate[-1] == min(self.data.funding_rate[-5:]))
            trend_condition = self.sma50[-1] > self.sma50[-5]  # Prevent entry in strong downtrend
            
            if cmo_condition and funding_condition and trend_condition:
                # Calculate risk parameters with cosmic precision
                entry_price = self.data.Close[-1]
                sl_price = min(self.swing_low[-1], entry_price * 0.98)
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = entry_price - sl_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        # Execute entry with OCO orders (Orbital Command Orders)
                        self.buy(
                            size=position_size,
                            sl=sl_price,
                            tp=self.upper_band[-1],
                            tag="🌙✨ ContrarianVolt Entry"
                        )
                        print(f"🚀🌕 MOON DEV ALERT: Cosmic Long Entry @ {entry_price:.2f}")
                        print(f"⚡ Risk: {risk_amount:.2f}, Size: {position_size}, SL: {sl_price:.2f}, TP: {self.upper_band[-1]:.2f}")

# Run backtest with lunar energy
print("\n🌑🌒🌓🌔🌕🌖🌗 MOON DEV BACKTEST INITIATION SEQUENCE 🌗🌖🌕🌔🌓🌒🌑")
bt = Backtest(data, ContrarianVolt, cash=1_000_000, exclusive_orders=True)
stats = bt.run()
print("\n🌙🌙🌙🌙🌙🌙🌙 MOON DEV BACKTEST RESULTS 🌙🌙🌙🌙🌙🌙🌙")
print(stats)
print(stats._strategy)