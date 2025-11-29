I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# DATA PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping to backtesting.py format
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data = data.rename(columns=column_map).set_index(pd.to_datetime(data['datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

class SqueezeMomentumBreakout(Strategy):
    bb_period = 20
    cmo_period = 14
    fi_period = 13
    squeeze_threshold = 0.015
    position_size = 0.02  # 2% of equity per trade
    max_positions = 5
    
    def init(self):
        # 🌗 INDICATOR CALCULATIONS
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, self.bb_period, 2, 2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_UPPER')
        
        self.bb_middle = self.I(talib.SMA, self.data.Close, self.bb_period, name='BB_MID')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, self.bb_period, 2, 2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_LOWER')
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        
        # Elder Force Index System
        def force_index(close, volume):
            fi = (close - pd.Series(close).shift(1)) * volume
            return talib.EMA(fi, self.fi_period)
        self.fi = self.I(force_index, self.data.Close, self.data.Volume, name='FORCE_INDEX')
        self.fi_sma = self.I(talib.SMA, self.fi, 5, name='FI_SMA')
        
        # Daily tracking
        self.current_date = None
        self.daily_equity = None
        self.trading_enabled = True

    def next(self):
        # 🌓 DAILY RISK MANAGEMENT
        current_date = self.data.index[-1].date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_equity = self.equity
            self.trading_enabled = True
            print(f"🌕 NEW DAY {current_date} | Starting Equity: ${self.daily_equity:,.2f}")
            
        if not self.trading_enabled:
            return
            
        # Check 5% daily loss limit
        if (self.equity - self.daily_equity) / self.daily_equity <= -0.05:
            print(f"🌧️ DAILY LOSS LIMIT TRIGGERED! Closing all positions.")
            self.position.close()
            self.trading_enabled = False
            return
            
        # 🌟 ENTRY LOGIC
        if len(self.trades) < self.max_positions and not self.position and self.trading_enabled:
            # Time filter (first 2 hours)
            hour = self.data.index[-1].hour
            if hour >= 2:
                return
                
            # Check last 3 bars for setup
            squeeze_condition = False
            cmo_condition = False
            breakout_condition = False
            
            for i in range(3):
                if len(self.data.Close) < self.bb_period + i:
                    continue
                    
                # Squeeze check
                width = (self.bb_upper[-i] - self.bb_lower[-i]) / self.bb_middle[-i]
                if width <= self.squeeze_threshold:
                    squeeze_condition =