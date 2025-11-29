Here's the fully debugged and completed code with Moon Dev's cosmic touch ✨🌙:

```python
# 🌙 Moon Dev's LiquidationBandwidth Backtest 🌙
from backtesting import Strategy, Backtest
import pandas as pd
import talib
import numpy as np

# Data preparation
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Rename columns with proper case
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    # Convert datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

# Custom indicator function for BBANDS and Bandwidth
def bbands_and_bandwidth(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(
        close, 
        timeperiod=timeperiod,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=0
    )
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth

class LiquidationBandwidthStrategy(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (fractional equity)
    bandwidth_lookback = 960  # 10 days in 15m intervals (10*24*4=960)
    swing_period = 20  # Swing high/low lookback
    
    def init(self):
        # Calculate indicators with proper TA-Lib integration
        self.upper_band, self.middle_band, self.lower_band, self.bandwidth = self.I(
            bbands_and_bandwidth,
            self.data.Close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            name='BBANDS_BW'
        )
        
        # Volatility contraction filter
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, self.bandwidth_lookback, name='BW_MIN')
        
        # Liquidation clusters (swing levels as proxy)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='SWING_LOW')
        
        # Track entry conditions
        self.entry_bandwidth = None

    def next(self):
        # Moon Dev's cosmic risk management 🌌
        equity = self.equity
        position_size = 0
        
        # Exit conditions first
        if self.position:
            if self.position.is_long:
                # Volatility expansion exit
                if self.bandwidth[-1] > 2 * self.entry_bandwidth:
                    self.position.close()
                    print(f"🌕✨ Volatility explosion! Closing LONG at {self.data.Close[-1]:.2f}")
                # Cluster flip exit
                elif self.data.Close[-1] < self.swing_low[-1]:
                    self.position.close()
                    print(f"🌑🔥 Cluster flip detected! Closing LONG at {self.data.Close[-1]:.2f}")
            
            elif self.position.is_short:
                # Volatility expansion exit
                if self.bandwidth[-1] > 2 * self.entry_bandwidth:
                    self.position.close()
                    print(f"🌕✨ Volatility explosion! Closing SHORT at {self.data.Close[-1]:.2f}")
                # Cluster flip exit
                elif self.data.Close[-1] > self.swing_high[-1]:
                    self.position.close()
                    print(f"🌑🔥 Cluster flip detected! Closing SHORT at {self.data.Close[-1]:.2f}")
                    
            return

        # Entry conditions - Cosmic alignment check 🌠
        if self.bandwidth[-1] <= self.bandwidth_min[-1]:
            # Long entry
            if self.data.High[-1] > self.swing_high[-1]:
                risk_amount = equity * self.risk_per_trade
                entry_price = self.swing_high[-1]
                stop_loss = self.swing_low[-1]
                
                if entry_price <= stop_loss:
                    print("🚨 Invalid risk parameters! Cosmic alignment failed 🌌")
                    return
                
                risk_per_unit = entry_price - stop_loss
                position_size