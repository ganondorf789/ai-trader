```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

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
        self.bbw_low_ind = self.I(talib.MIN, self.bbw_ind, timeperiod=self.squeeze_period, name='BBW_LOW')
        
        # RSI (14-period)
        self.rsi_ind = self.I(talib.RSI, close, timeperiod=14, name='RSI')
        
        # Track entry bar for time exits
        self.entry_bar = 0
        
    def next(self):
        # Moon Dev Debug Prints 🌙💬
        print(f"🌙 Close: {self.data.Close[-1]:.2f} | BBW: {self.bbw_ind[-1]:.4f} | BBW Low: {self.bbw_low_ind[-1]:.4f} | RSI: {self.rsi_ind[-1]:.2f}")

        if self.position:
            # Time-based exit (3 candles)
            if len(self.data) - self.entry_bar >= 3:
                self.position.close()
                print(f"🌙✨ Time exit at bar {len(self.data)}! ⏳")
            
            # RSI exits
            if self.position.is_long and crossover(self.rsi_ind[-2:], 70):
                self.position.close()
                print(f"🌙✨ RSI EXIT LONG ({self.rsi_ind[-1]:.2f}) 📉")
            elif self.position.is_short and crossunder(self.rsi_ind[-2:], 30):
                self.position.close()
                print(f"🌙✨ RSI EXIT SHORT ({self.rsi_ind[-1]:.2f}) 📈")
        else:
            # Check squeeze condition (BBW at 10-day low)
            if self.bbw_ind[-1] <= self.bbw_low_ind[-1]:
                # Long entry logic
                if self.data.High[-1] > self.bb_upper_ind[-1]:
                    # Moon Dev Liquidation Confirmation (placeholder) 🌙📈
                    print("🌙✨ ORDER BOOK CONFIRMS LONG! 🚀")
                    self._enter_trade('long')
                
                # Short entry logic
                elif self.data.Low[-1] < self.bb_lower_ind[-1]:
                    # Moon Dev Liquidation Confirmation (placeholder) 🌙📉
                    print("🌙✨ ORDER BOOK CONFIRMS SHORT! 🚀")
                    self._enter_trade('short')

    def _enter_trade(self, direction):
        # Unified entry logic with Moon Dev Risk Management 🌙🛡️
        price = self.data.Close[-1]
        
        if direction == '