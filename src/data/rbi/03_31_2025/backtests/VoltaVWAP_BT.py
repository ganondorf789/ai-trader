```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaVWAP(Strategy):
    multiplier = 2
    risk_pct = 0.01
    vol_window = 20
    
    def init(self):
        # 🌙 Calculate VWAP with daily anchor
        self.vwap = self.I(ta.vwap,
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume,
            anchor='D',
            name='VWAP'
        )
        
        # ✨ Volatility calculations
        self.returns = self.I(lambda x: x.Close.pct_change(), name='Returns')
        self.volatility = self.I(talib.STDDEV, self.returns, self.vol_window, name='Volatility')
        
        # 🚀 Dynamic bands
        self.upper_band = self.I(
            lambda v, vol: v * (1 + self.multiplier * vol),
            self.vwap, self.volatility, name='UpperBand'
        )
        self.lower_band = self.I(
            lambda v, vol: v * (1 - self.multiplier * vol),
            self.vwap, self.volatility, name='LowerBand'
        )
        print("🌙✨ Moon Dev Indicators Initialized!")
        
    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # 🌙 Long entry condition
            if crossover(price, self.upper_band):
                self.enter_trade('long')
            # 🌙 Short entry condition    
            elif crossover(self.lower_band, price):
                self.enter_trade('short')
        else:
            self.manage_trades()
    
    def enter_trade(self, direction):
        entry_price = self.data.Close[-1]
        stop_loss = self.lower_band[-1] if direction == 'long' else self.upper_band[-1]
        
        # 🛑 Risk calculations
        risk_amount = self.equity * self.risk_pct
        risk_per_share = abs(entry_price - stop_loss)
        volatility = max(self.volatility[-1], 0.0001)  # Prevent division by zero
        
        # 🌙 Dynamic position sizing
        position_size = (risk_amount / risk_per_share) * (1/volatility)
        position_size = int(round(position_size))
        position_size = min(position_size, int(self.equity//entry_price))
        
        if position_size > 0:
            if direction == 'long':
                self.buy(size=position_size, sl=stop_loss)
                print(f"🚀🌙 LONG ENTRY | Price: {entry_price:.2f} | Size: {position_size}")
            else:
                self.sell(size=position_size, sl=stop_loss)
                print(f"🌙🚀 SHORT ENTRY | Price: {entry_price:.2f} | Size: {position_size}")
    
    def manage_trades(self):
        price = self.data.Close[-1]
        
        if self.position.is_long:
            # 🎯 Take profit at VWAP
            if crossover(self.vwap, price):
                self.position.close()
                print(f"🎯🌙 LONG TP | Closed at {price:.2f}")
            # 🛑 Stop loss at lower band
            elif price < self.lower_band[-1]:
                self.position.close()
                print(f"🛑🌙 LONG SL | Closed at {price:.2f}")
                
        elif self.position.is_short:
            # 🎯 Take profit at VWAP
            if crossover(price, self.vwap):
                self.position.close()
                print(f"🎯🌙 SHORT TP | Closed at {price:.2f}")
            # 🛑 Stop loss at upper band
            elif price > self.upper_band[-1]:
                self.position.close()
                print(f"🛑