I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandwidthSurge(Strategy):
    bb_period = 20
    bb_dev = 2
    bb_low_lookback = 17280  # 6 months in 15min intervals
    volume_ma_period = 2880  # 30 days in 15min
    atr_period = 14
    risk_pct = 0.01
    
    def init(self):
        # Calculate Bollinger Bands components
        close = self.data.Close
        self.upper, self.middle, self.lower = talib.BBANDS(close, 
            timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
        
        # Add indicators using proper self.I() wrapper
        self.bb_upper = self.I(lambda: self.upper, name='BB_Upper')
        self.bb_lower = self.I(lambda: self.lower, name='BB_Lower')
        self.bandwidth = self.I(lambda: (self.upper - self.lower)/self.middle, name='Bandwidth')
        
        # Volatility contraction detector
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, 
            timeperiod=self.bb_low_lookback, name='Bandwidth_Low')
        
        # Volume surge components
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 
            timeperiod=self.volume_ma_period, name='Volume_MA')
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
            timeperiod=self.atr_period, name='ATR')
        
        # Track trade parameters
        self.entry_price = None
        self.entry_atr = None
        self.entry_bandwidth = None

    def next(self):
        # Moon Dev debug prints 🌙
        print(f"🌙 Current Close: {self.data.Close[-1]:.2f} | "
              f"Bandwidth: {self.bandwidth[-1]:.4f} | "
              f"Volume Ratio: {self.data.Volume[-1]/self.volume_ma[-1]:.2f}x | "
              f"Equity: {self.equity:,.2f}")
        
        if not self.position:
            # Long entry condition
            if (self.bandwidth[-1] == self.bandwidth_low[-1] and
                self.data.Volume[-1] >= 1.5 * self.volume_ma[-1] and
                self.data.Close[-1] > self.bb_upper[-1]):
                
                # Calculate position size
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / self.atr[-1]))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.entry_atr = self.atr[-1]
                    self.entry_bandwidth = self.bandwidth[-1]
                    print(f"🚀🌙 LONG ENTRY! Size: {position_size} | "
                          f"Entry: {self.entry_price:.2f} | "
                          f"ATR: {self.entry_atr:.2f} | "
                          f"Bandwidth: {self.entry_bandwidth:.4f}")
            
            # Short entry condition    
            elif (self.bandwidth[-1] == self.bandwidth_low[-1] and
                  self.data.Volume[-1] >= 1.5 * self.volume_ma[-1] and
                  self.data.Close[-1] < self.bb_lower[-