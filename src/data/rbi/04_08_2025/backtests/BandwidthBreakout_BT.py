```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', 
                   parse_dates=['datetime'], 
                   index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandwidthBreakout(Strategy):
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        # Bollinger Bands and Bandwidth
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2),
            self.data.Close,
            name=['UpperBB', 'MiddleBB', 'LowerBB']
        )
        
        # Bandwidth calculation
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower,
            name='Bandwidth'
        )
        
        # 10-period Bandwidth Low
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, timeperiod=10, name='Bandwidth_Low')
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14, name='CMO')
        
        # Average True Range
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')

    def next(self):
        # Moon Dev Risk Management Parameters
        risk_pct = 0.01  # 1% risk per trade
        
        # Check if we're not in a position
        if not self.position:
            cmo_now = self.cmo[-1]
            cmo_prev = self.cmo[-2] if len(self.cmo) > 1 else 0
            bandwidth_now = self.bandwidth[-1]
            bandwidth_low_now = self.bandwidth_low[-1]

            # Long Entry Condition
            if (bandwidth_now == bandwidth_low_now and 
                cmo_prev <= 0 and cmo_now > 0):
                
                atr_value = self.atr[-1]
                if atr_value == 0:
                    return
                
                # Position sizing calculation
                risk_amount = risk_pct * self.equity
                position_size = int(round(risk_amount / (2 * atr_value)))
                
                if position_size > 0:
                    entry_price = self.data.Close[-1]
                    sl = entry_price - 2 * atr_value
                    self.buy(size=position_size, 
                            sl=sl,
                            tag={'extreme_price': self.data.High[-1]})
                    print(f"🌙 MOON DEV LONG ENTRY 🚀 Price: {entry_price:.2f}, Size: {position_size}, SL: {sl:.2f}")

            # Short Entry Condition    
            elif (bandwidth_now == bandwidth_low_now and 
                  cmo_prev >= 0 and cmo_now < 0):
                
                atr_value = self.atr[-1]
                if atr_value == 0:
                    return
                
                risk_amount = risk_pct * self.equity
                position_size = int(round(risk_amount / (2 * atr_value)))
                
                if position_size > 0:
                    entry_price = self.data.Close[-1]
                    sl = entry_price + 2 * atr_value
                    self.sell(size=position_size,
                             sl=sl,
                             tag={'extreme_price': self.data.Low[-1]})
                    print(f"🌙 MOON DEV SHORT ENTRY 🌑 Price: {entry_price:.2f}, Size: {position_size}, SL: {sl:.2f}")

        # Update trailing stops for open positions
        for trade in self.trades:
            atr_value = self.atr[-1]
            if trade.is_long:
                current_high = self.data.High[-1]
                if current_high > trade.tag['extreme_price']:
                    trade.tag['extreme_price