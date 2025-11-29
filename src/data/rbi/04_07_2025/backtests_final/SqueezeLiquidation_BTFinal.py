Here's the fully debugged and Moon Dev themed version of your SqueezeLiquidationStrategy:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙✨
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper case column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Convert datetime and set index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class SqueezeLiquidationStrategy(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    squeeze_period = 960  # 10 days in 15m candles (96*10)
    
    def init(self):
        # Moon Dev Indicator Setup 🌙📊
        close = self.data.Close
        
        # Bollinger Bands (20-period SMA, 2 std)
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Add BB indicators using self.I()
        self.bb_upper_ind = self.I(lambda: self.bb_upper, name='BB_UPPER')
        self.bb_lower_ind = self.I(lambda: self.bb_lower, name='BB_LOWER')
        
        # Bollinger Bandwidth (BBW)
        bbw = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bbw_ind = self.I(lambda: bbw, name='BBW')
        
        # 10-day BBW low (960-period MIN)
        self.bbw_low_ind = self.I(lambda: talib.MIN(bbw, timeperiod=self.squeeze_period), name='BBW_LOW')
        
        # RSI (14-period)
        self.rsi_ind = self.I(talib.RSI, close, timeperiod=14, name='RSI')
        
        # Track entry bar for time exits
        self.entry_bar = 0
        
    def _enter_trade(self, direction):
        """Moon Dev Trade Entry Function 🌙💼"""
        if direction == 'long':
            stop_price = self.bb_lower_ind[-1]
            risk_per_trade = self.data.Close[-1] - stop_price
            size = (self.equity * self.risk_percent) / risk_per_trade
            size = min(size, 1)  # Cap at 100% of equity
            self.buy(size=size, sl=stop_price)
            print(f"🌙✨ LONG ENTRY at {self.data.Close[-1]:.2f} | Size: {size:.2%} | SL: {stop_price:.2f} 🚀")
        elif direction == 'short':
            stop_price = self.bb_upper_ind[-1]
            risk_per_trade = stop_price - self.data.Close[-1]
            size = (self.equity * self.risk_percent) / risk_per_trade
            size = min(size, 1)  # Cap at 100% of equity
            self.sell(size=size, sl=stop_price)
            print(f"🌙✨ SHORT ENTRY at {self.data.Close[-1]:.2f} | Size: {size:.2%} | SL: {stop_price:.2f} 📉")
        self.entry_bar = len(self.data)
        
    def next(self):
        # Moon Dev Debug Prints 🌙💬
        print(f"🌙 Close: {self.data.Close[-1]:.2f} | BBW: {self.bbw_ind[-1]:.4f} | BBW Low: {self.bbw_low_ind[-1]:.4f} | RSI: {self.rsi_ind[-1]:.2f}")

        if self.position:
            # Time-based exit (3 candles)
            if len(self.data) - self.entry_bar >= 3:
                self.position.close()
                print(f"🌙✨ Time exit at bar {len(self.data)}! ⏳")
            
            # RSI exits - replaced crossovers with direct comparisons
            if self.position.is_long