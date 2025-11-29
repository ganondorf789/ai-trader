Here's the implementation of the MomentumRipple strategy for backtesting.py:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🚀 DATA PREPARATION 
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🌌 CLEANSE COSMIC DATA
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class MomentumRipple(Strategy):
    # 🌠 STRATEGY CONSTELLATIONS
    rmi_length = 14
    rmi_momentum = 5
    entry_threshold = 70  # R1 activation level
    lookback_weeks = 52
    risk_pct = 0.7  # 70% position sizing
    stop_loss = 0.2  # 20% stop
    
    def init(self):
        # 🌗 PHASE 1: LONG-TERM MOMENTUM FILTER
        lookback = self.lookback_weeks * 7 * 24 * 4  # 52 weeks in 15m
        self.price_filter = self.I(lambda close: 
            ((close - close.shift(lookback)) / close.shift(lookback)) * 100,
            self.data.Close
        )
        
        # 🌓 PHASE 2: RELATIVE MOMENTUM INDEX
        self.rmi = self.I(ta.rmi, self.data.Close, 
                         length=self.rmi_length, 
                         momentum=self.rmi_momentum)
        
        # 🌍 PHASE 3: RMI MOVEMENT TRACKING
        self.rmi_changes = self.I(lambda x: x.diff(), self.rmi)
        
        # 🛰️ POSITION TRACKING VARIABLES
        self.entry_rmi_value = None
        self.cumulative_gain = 0

    def next(self):
        # 🌕 CURRENT COSMIC READINGS
        price_filter = self.price_filter[-1]
        current_rmi = self.rmi[-1]
        prev_rmi = self.rmi[-2] if len(self.rmi) > 1 else 0
        
        if self.position:
            # 🚨 RISK MANAGEMENT PROTOCOLS
            current_price = self.data.Close[-1]
            
            # 🌧️ STOP-LOSS CHECK
            if current_price <= self.position.entry_price * (1 - self.stop_loss):
                self.position.close()
                print(f"🌧️ MOON ALERT! Stop-Loss Activated at {current_price:.2f}")
                return
                
            # 🛸 CUMULATIVE GAIN TRACKING
            rmi_change = self.rmi_changes[-1]
            if rmi_change > 0:
                self.cumulative_gain += rmi_change
                print(f"📈 Ripple Gain: +{rmi_change:.2f} | Total: {self.cumulative_gain:.2f}")
                
            # 🎯 EXIT CONDITION CHECK
            if self.cumulative_gain >= 3 * self.entry_rmi_value:
                self.position.close()
                print(f"🚀 STARSHIP EXIT! Cumulative Gain {self.cumulative_gain:.2f} "
                      f"3x Initial RMI {self.entry_rmi_value:.2f}")
        else:
            # 🌟 ENTRY CONDITIONS
            if (price_filter > 0 and 
                current_rmi >= self.entry_threshold and 
                prev_rmi < self.entry_threshold):
                
                # 🧮 POSITION SIZING CALCULATION
                equity = self.equity()
                position_size = (equity * self.risk_pct) / self.data.Close[-1]
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_rmi_value = current_rmi
                    self.cumulative_gain = 0
                    print(f"🌙 MOONSHOT ACTIVATED! Entry RMI: {current_rmi:.2f} "