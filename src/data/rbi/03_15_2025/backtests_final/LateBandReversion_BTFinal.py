I'll help debug and fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import time

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
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

class LateBandReversion(Strategy):
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    risk_pct = 0.01
    
    def init(self):
        # Bollinger Bands using talib
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, 
                                                    self.data.Close, 
                                                    timeperiod=self.bb_period,
                                                    nbdevup=self.bb_dev,
                                                    nbdevdn=self.bb_dev,
                                                    matype=0)
        
        # ATR using talib
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         timeperiod=self.atr_period)
        
        # Initialize trade counts and session tracking
        self.trade_count = {'long': 0, 'short': 0}
        self.current_date = None
        self.in_session = False

    def next(self):
        current_dt = self.data.index[-1]
        new_date = current_dt.date()
        
        # Reset trade counts at midnight
        if new_date != self.current_date:
            self.trade_count = {'long': 0, 'short': 0}
            self.current_date = new_date
            print(f"🌙🔄 Moon Dev Reset: Trade counts cleared for new day {new_date}")
            
        # Check late session hours (22:00-23:59 UTC)
        self.in_session = current_dt.hour in [22, 23]
        
        # Close positions at session end
        if not self.in_session and self.position:
            self.position.close()
            print(f"🌙🕒 Moon Dev Session Close: Position closed at {self.data.Close[-1]:.2f}")
            return
            
        if not self.in_session:
            return
            
        # Calculate risk parameters
        equity = self.equity()
        atr_value = self.atr[-1]
        stop_distance = 1.5 * atr_value
        risk_amount = equity * self.risk_pct
        
        # Entry logic
        if not self.position:
            # Short entry condition
            if self.data.High[-1] > self.upper[-1] and self.trade_count['short'] < 1:
                if stop_distance == 0:
                    print("🌙⚠️ Moon Dev Warning: Stop distance is zero, skipping trade")
                    return
                size = int(round(risk_amount / stop_distance))
                if size == 0:
                    print("🌙⚠️ Moon Dev Warning: Calculated size is zero, skipping trade")
                    return
                entry_price = self.data.Close[-1]
                sl_price = entry_price + stop_distance
                
                self.sell(size=size, 
                         sl=sl_price,
                         tag=f'Short | SL: {sl_price:.2f}')
                self.trade_count['short'] += 1
                print(f"🌙🚀 SHORT ENTRY! {entry_price:.2f} | Size: {size} BTC | SL: {sl_price:.2f}")
            
            # Long entry condition
            elif self.data.Low[-1] < self.lower[-1] and self.trade_count['long'] < 1:
                if stop_distance == 0:
                    print("🌙⚠️ Moon Dev Warning: Stop distance is zero, skipping trade")
                    return
                size = int(round(risk_amount / stop_distance))
                if size == 0:
                    print("🌙⚠️ Moon Dev Warning: Calculated size is zero, skipping trade")
                    return
                entry_price = self.data.Close[-1]
                sl_price = entry_price - stop_distance
                
                self.buy(size=size,
                        sl=sl_price,
                        tag=f'Long | SL: {sl_price:.2