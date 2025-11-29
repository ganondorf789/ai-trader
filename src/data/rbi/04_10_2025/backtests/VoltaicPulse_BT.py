import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pytz

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data = data.rename(columns=column_map)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.index = data.index.tz_localize('UTC')  # Assume original data is UTC

class VoltaicPulse(Strategy):
    timeperiod_atr = 14
    timeperiod_ema = 20
    vix_sma_period = 960  # 10 days in 15m intervals (10*24*4=960)
    risk_pct = 0.01
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.timeperiod_atr)
        self.atr_ema = self.I(talib.EMA, self.atr, self.timeperiod_ema)
        self.vix_sma = self.I(talib.SMA, self.data['vix'], self.vix_sma_period)
        
        # Initialize trailing stop variables
        self.highest_high = None
        
    def next(self):
        # Convert current time to Eastern Time
        current_time = self.data.index[-1].astimezone(pytz.timezone('US/Eastern'))
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check market hours
        if not (market_open.time() <= current_time.time() <= market_close.time()):
            print(f"🌙 Moon Dev: Skipping non-market hours {current_time.strftime('%Y-%m-%d %H:%M:%S%z')} 🕒")
            return
            
        # Entry logic
        if not self.position:
            # Check ATR crossover condition
            if (crossover(self.atr, self.atr_ema) and \
               (self.data['vix'][-1] < self.vix_sma[-1]):
                
                # Calculate position size with risk management
                atr_value = self.atr[-1]
                stop_distance = 2 * atr_value
                position_size = (self.risk_pct * self.equity) / stop_distance
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🚀 MOON DEV ALERT: Long entry at {self.data.Close[-1]:.2f}!")
                    print(f"   Size: {position_size} units | ATR: {atr_value:.2f} 🌕")
        
        # Exit logic with trailing stop
        else:
            current_high = self.data.High[-1]
            self.highest_high = max(self.highest_high, current_high) if self.highest_high else current_high
            
            # Calculate dynamic trailing stop
            current_atr = self.atr[-1]
            stop_level = self.highest_high - 2 * current_atr
            
            if self.data.Low[-1] <= stop_level:
                self.position.close()
                print(f"🌑 MOON DEV EXIT: Trailing stop triggered at {stop_level:.2f}!")
                print(f"   Profit: {self.position.pl_pct:.2%} ✨")
                self.highest_high = None

# Run backtest
bt = Backtest(data, VoltaicPulse, cash=1_000_000, margin=1.0)
stats = bt.run()
print(stats)
print(stats._strategy)