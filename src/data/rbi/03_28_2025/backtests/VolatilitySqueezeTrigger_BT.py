```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and clean data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySqueezeTrigger(Strategy):
    risk_per_trade = 0.02
    atr_multiplier = 2
    
    def init(self):
        # Calculate indicators using TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_bb = self.I(lambda: upper, name='UpperBB')
        self.lower_bb = self.I(lambda: lower, name='LowerBB')
        
        # Historical Volatility (20-period STDDEV)
        self.hv = self.I(talib.STDDEV, close, timeperiod=20, nbdev=1)
        
        # Average True Range
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        
        # Trackers
        self.peak_equity = self._broker.equity
        self.entry_price = 0
        self.highest_high = 0
        self.lowest_low = 0

    def next(self):
        current_idx = len(self.data) - 1
        
        # Check drawdown limit
        current_equity = self._broker.equity
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity)/self.peak_equity
        if drawdown >= 0.10:
            print(f"🌧️ MOON DEV CRISIS: 10% Drawdown - Trading Halted!")
            return

        if not self.position:
            if current_idx < 20:
                return
                
            # Check squeeze condition (previous bar)
            prev_upper = self.upper_bb[-2]
            prev_lower = self.lower_bb[-2]
            bb_width = prev_upper - prev_lower
            prev_hv = self.hv[-2]
            squeeze = bb_width < prev_hv
            
            # Get previous bar values
            prev_close = self.data.Close[-2]
            prev_atr = self.atr[-2]
            
            # Long entry logic
            if squeeze and (prev_close > prev_upper + self.atr_multiplier * prev_atr):
                risk_amount = self.risk_per_trade * current_equity
                risk_per_share = self.atr_multiplier * prev_atr
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=self.data.Open[-1] - 2*prev_atr)
                    print(f"🚀🌙 MOON DEV LONG LAUNCH: {self.data.Open[-1]:.2f}, Size: {position_size}")
            
            # Short entry logic
            elif squeeze and (prev_close < prev_lower - self.atr_multiplier * prev_atr):
                risk_amount = self.risk_per_trade * current_equity
                risk_per_share = self.atr_multiplier * prev_atr
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.sell(size=position_size, sl=self.data.Open[-1] + 2*prev_atr)
                    print(f"🌑🌙 MOON DEV SHORT BLAST: {self.data.Open[-1]:.2f}, Size: {position_size}")
        
        else:
            # Manage open positions
            current_close = self.data.Close[-1]
            current_atr = self.atr[-1]
            
            if self.position.is_long:
                # Exit conditions
                if (current_close < self.position.entry_price - 2*current_at