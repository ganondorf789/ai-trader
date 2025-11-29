```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Strategy, Backtest
import talib

class LiquidityReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators with self.I()
        self.prev_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='5-period High')
        self.prev_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='5-period Low')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR(14)')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=3, name='Volume SMA(3)')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR MA(20)')
        
        self.entry_bar = 0  # Track entry bar for time-based exit

    def next(self):
        current_bar = len(self.data) - 1
        
        if self.position:
            # Check time-based exit (4 hours = 16*15m bars)
            if current_bar - self.entry_bar >= 16:
                self.position.close()
                print(f"🌙✨ Time-based exit at bar {current_bar}! Moon Dev's patience expired!")
            return
        
        # Get current values
        close = self.data.Close[-1]
        prev_high = self.prev_high[-1]
        prev_low = self.prev_low[-1]
        volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        atr_val = self.atr[-1]
        atr_ma_val = self.atr_ma[-1]
        
        # Check volatility filter
        if atr_val <= atr_ma_val:
            return
        
        # Entry conditions
        if close > prev_high and volume < volume_sma:
            # 🌙 LONG Entry Logic
            risk_amount = self.risk_percent * self._broker.equity
            stop_distance = 2 * atr_val
            position_size = risk_amount / stop_distance
            position_size = int(round(position_size))
            
            if position_size > 0:
                midpoint_tp = (prev_high + prev_low) / 2
                sl_price = close - stop_distance
                self.buy(size=position_size, sl=sl_price, tp=midpoint_tp)
                self.entry_bar = current_bar
                print(f"🚀🌕 LONG Signal! Buying {position_size} units at {close:.2f}")
                print(f"   TP: {midpoint_tp:.2f} | SL: {sl_price:.2f} | Moon Rocket Engaged!")
                
        elif close < prev_low and volume < volume_sma:
            # 🌑 SHORT Entry Logic
            risk_amount = self.risk_percent * self._broker.equity
            stop_distance = 2 * atr_val
            position_size = risk_amount / stop_distance
            position_size = int(round(position_size))
            
            if position_size > 0:
                midpoint_tp = (prev_high + prev_low) / 2
                sl_price = close + stop_distance
                self.sell(size=position_size, sl=sl_price, tp=midpoint_tp)
                self.entry_bar = current_bar
                print(f"🌘🌑 SHORT Signal! Selling {position_size} units at {close:.2f}")
                print(f"   TP: {midpoint_tp:.2f} | SL: {sl_price:.2f} | Surfing Moon Gravity!")

# Data preparation
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',