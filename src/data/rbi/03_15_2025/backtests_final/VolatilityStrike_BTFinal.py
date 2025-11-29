I've analyzed the code and made the necessary technical fixes while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's VolatilityStrike Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# 🚀 DATA PREPARATION
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityStrike(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌙
    
    def init(self):
        # ✨ INDICATORS (using talib)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.entry_atr = None
        self.entry_bar = None
        self.entry_price = None
        self.entry_risk = None

    def next(self):
        current_idx = len(self.data) - 1
        
        if not self.position:
            # 🌙 ENTRY LOGIC
            if (self.data.High[-1] == self.swing_high[-1] and 
                (self.atr[-1] / self.data.Close[-1]) * 100 >= 2):
                
                # 🚀 RISK MANAGEMENT
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.data.Close[-1] * 1.01  # 1% stop
                take_profit = self.data.Close[-1] * 0.98  # 2% target
                
                # 🌙 Position sizing fix
                risk_per_unit = abs(stop_loss - self.data.Close[-1])
                position_size = int(round(risk_amount / risk_per_unit))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    self.entry_atr = self.atr[-1]
                    self.entry_bar = current_idx
                    self.entry_price = self.data.Close[-1]
                    self.entry_risk = risk_amount
                    
                    print(f"🌙✨ ENTRY: Short {position_size} units at {self.data.Close[-1]:.2f}")
                    print(f"   ATR%: {(self.atr[-1] / self.data.Close[-1]) * 100:.2f}% 🚀")
                    print(f"   Stop: {stop_loss:.2f} | Target: {take_profit:.2f} 🌙")
        else:
            # 🌙 EXIT CONDITIONS
            days_in_trade = (current_idx - self.entry_bar) / 96  # 96=15m bars/day
            
            # IV Drop Condition
            if (self.entry_atr - self.atr[-1]) / self.entry_atr >= 0.25:
                self.position.close()
                print(f"🌙📉 EXIT: IV dropped 25%! Closed at {self.data.Close[-1]:.2f}")
                print(f"   P/L: {self.position.pl:.2f} | Days held: {days_in_trade:.2f} 🌙")
                
            # Time Exit (1 day before 7-day expiration)
            elif days_in_trade >= 6:
                self.position.close()
                print(f"🌙⏳ EXIT: 6 days held! Closed at {self.data.Close[-1]:.2f}")
                print(f"   P/L: {self.position.pl:.2f} 🌙")
                
            # Stop Loss (50% of risk)
            elif self.position.pl <= -0.5 * self.entry_risk:
                self.position.close()
                print(f"🌙💥 EXIT: 50% loss hit! Closed at {self.data.Close[-1]:.2f}")
                print(f"   P/L: {self.position.pl:.2f} 🌙")

# 🚀 BACKTEST EXECUTION
bt