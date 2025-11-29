I'll fix the code by removing the `backtesting.lib` import and replacing any usage of its functions with native Python/numpy/pandas alternatives. Here's the corrected version:

```python
# 🌙 MOON DEV BACKTESTING SUITE :: LiquidationBounce Strategy 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ===================
# DATA PREPARATION
# ===================
# Load and clean market data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🌙 Cleanse the cosmic data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌌 Align celestial columns
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ===================
# STRATEGY CLASS
# ===================
class LiquidationBounce(Strategy):
    entry_bar = 0
    _consecutive_losses = 0  # Track cosmic misfortune 🌑
    
    def init(self):
        # 🌗 Calculate Bollinger Bands using TA-Lib
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['UPPER_BB', 'MIDDLE_BB', 'LOWER_BB']
        )
        
        # 🌀 Liquidations MA calculation
        self.liquidation_ma = self.I(talib.SMA, 
            self.data.df['liquidation'],  # Cosmic liquidation data
            timeperiod=30, name='LIQUIDATION_MA'
        )

    def next(self):
        # 🌓 Need at least 2 bars for calculations
        if len(self.data.Close) < 2:
            return
            
        prev_low = self.data.Low[-2]
        prev_lower_bb = self.lower_bb[-2]
        prev_liq = self.data.df['liquidation'][-2]
        prev_liq_ma = self.liquidation_ma[-2]

        # 🚀 Entry Conditions
        if not self.position:
            # Cosmic alignment check
            bb_touch = prev_low <= prev_lower_bb
            liq_trigger = prev_liq > prev_liq_ma
            
            if bb_touch and liq_trigger:
                # 🌕 Calculate moon mission parameters
                risk_pct = 0.02  # 2% cosmic risk limit
                account_equity = self.equity
                risk_amount = account_equity * risk_pct
                
                entry_price = self.data.Open[-1]
                stop_loss = entry_price - 1.5*(entry_price - prev_lower_bb)
                risk_per_unit = entry_price - stop_loss
                
                if risk_per_unit <= 0:
                    print("🌑 ABORT: Negative risk detected!")
                    return
                
                # 🪐 Calculate position size
                position_size = int(round(risk_amount / risk_per_unit))
                
                if position_size <= 0:
                    print(f"🌒 Tiny position: {position_size} units")
                    return
                
                if self._consecutive_losses >= 3:
                    print("🌑 COOLDOWN: 3 consecutive losses")
                    return
                
                # 🚀 Execute moon mission
                self.buy(
                    size=position_size,
                    sl=stop_loss,
                    tp=self.middle_bb[-1]  # Middle BB as TP
                )
                self.entry_bar = len(self.data) -1
                
                print(f"\n🚀 MOON SHOT! 🚀"
                      f"\nEntry: {entry_price:.2f}"
                      f"\nSize: {position_size} units"
                      f"\nSL: {stop_loss:.2f}"
                      f"\nTP: {self.middle_bb[-1]:.2f}")

        # 🌗 Time-based exit
        if self.position and (len(self.data)-1 - self.entry_bar) >= 5:
            self.position.close()
            print(f"🌑 TIME EXIT after 5 bars")

    def notify