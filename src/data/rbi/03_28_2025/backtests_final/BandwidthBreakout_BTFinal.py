I'll complete and fix the code while maintaining the original strategy logic. Here's the fully debugged version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's BandwidthBreakout Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class BandwidthBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌟 Core Indicators 🌟
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS,
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_UPPER', 'BB_MID', 'BB_LOWER']
        )
        
        # 🔍 Bandwidth Calculation
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.I(lambda x: x, self.bb_width, name='BB_WIDTH')
        
        # 📉 20-day Bandwidth Low
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=20, name='BB_WIDTH_LOW')
        
        # 🔊 Volume Filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=10, name='VOL_SMA_10')
        
        # 🛑 ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR_14')

    def next(self):
        # 🕒 Wait for indicators to warm up
        if len(self.data) < 20:
            return
        
        # 🌙 Entry Logic
        if not self.position:
            # Long Entry Conditions 🚀
            if (self.bb_width[-1] == self.bb_width_low[-1] and
                self.data.Close[-1] > self.bb_upper[-1] and
                self.data.Volume[-1] > 1.5 * self.volume_sma[-1]):
                
                # Risk Management Calculations 🌙
                atr = self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = 2 * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🌙✨ MOON DEV LONG ENTRY @ {self.data.Close[-1]:.2f} | Size: {position_size} BTC 🚀")
            
            # Short Entry Conditions 🌌
            elif (self.bb_width[-1] == self.bb_width_low[-1] and
                  self.data.Close[-1] < self.bb_lower[-1] and
                  self.data.Volume[-1] > 1.5 * self.volume_sma[-1]):
                
                # Risk Management Calculations 🌙
                atr = self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = 2 * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    print(f"🌙✨ MOON DEV SHORT ENTRY @ {self.data.Close[-1]:.2f} | Size: {position_size} BTC 🌌")
        
        # 🌑 Exit Logic
        else:
            for trade in self.trades:
                if trade.is_long:
                    # Long Exit: 2x ATR below highest high since entry 🌙
                    highest_high = max(self.data.High[trade.entry_bar:len(self.data)])
                    exit_price = highest_high