Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed enhancements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names and handle data ✨
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

# Calculate daily ATR and merge back 🌕
daily_df = data.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
daily_high = daily_df['High'].values
daily_low = daily_df['Low'].values
daily_close = daily_df['Close'].values
daily_atr = talib.ATR(daily_high, daily_low, daily_close, 14)
daily_df['Daily_ATR'] = daily_atr.shift(1)
data = data.merge(daily_df['Daily_ATR'], left_index=True, right_index=True, how='left')
data['Daily_ATR'] = data['Daily_ATR'].ffill()

class VolatilitySurge(Strategy):
    def init(self):
        # Core indicators 🌙✨
        self.atr_15m = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_15m')
        
        # Bollinger Bands with TA-Lib 🌗
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        self.trailing_stop = None
        self.highest_high = None

    def next(self):
        # Moon Dev monitoring system 🌙📊
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Pulse | Equity: ${self.equity:,.2f} | Price: {self.data.Close[-1]:.2f} ✨")
        
        if not self.position:
            # Entry logic 🚀🌕
            atr_condition = self.atr_15m[-1] > 2 * self.data.Daily_ATR[-1]
            bb_condition = self.data.Close[-1] > self.bb_upper[-1]
            
            if atr_condition and bb_condition:
                # Risk management 🌗🛡️
                risk_percent = 0.01
                entry_price = self.data.Close[-1]
                atr_trail = self.atr_15m[-1] * 1.5
                stop_loss = entry_price - atr_trail
                risk_amount = self.equity * risk_percent
                position_size = int(round(risk_amount / (entry_price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.trailing_stop = stop_loss
                    self.highest_high = entry_price
                    print(f"🚀🌕 MOON SHOT! Long {position_size} @ {entry_price:.2f} | SL: {stop_loss:.2f} 🌕🚀")
        else:
            # Update trailing stop 🌑📉
            self.highest_high = max(self.highest_high, self.data.High[-1])
            new_stop = self.highest_high - (self.atr_15m[-1] * 1.5)
            self.trailing_stop = max(self.trailing_stop, new_stop)
            
            # Exit logic 🛑🌑
            if self.data.Low[-1] < self.trailing_stop:
                self.position.close()
                print(f"🌑💫 MOON OUT! Exit @ {self.data.Close[-1]:.2f} |