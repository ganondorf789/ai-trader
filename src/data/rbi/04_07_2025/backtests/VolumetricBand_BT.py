```python
# 🌙 Moon Dev's VolumetricBand Backtest 🌙
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import os
os.environ['MPLBACKEND'] = 'Agg'  # Disable GUI

# Data preparation function
def prepare_data(filepath):
    data = pd.read_csv(filepath)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Rename columns to proper case
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=column_map)
    # Convert datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

class VolumetricBand(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_concurrent_trades = 3
    
    def init(self):
        # 🌙 Indicator Declaration
        # Bollinger Bands
        def bb_calculation(close):
            upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper, lower
        self.bb_upper, self.bb_lower = self.I(bb_calculation, self.data.Close)
        
        # VWAP (using pandas_ta)
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume)
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Daily drawdown tracking
        self.current_day = None
        self.daily_high_equity = self._broker.cash

    def next(self):
        # 🌙 Moon Dev's Daily Risk Check
        current_date = self.data.index[-1].date()
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_high_equity = self.equity
        else:
            self.daily_high_equity = max(self.daily_high_equity, self.equity)
        
        current_drawdown = (self.daily_high_equity - self.equity) / self.daily_high_equity
        block_entries = current_drawdown > 0.05
        
        # 🚀 Trade Logic
        close_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        volume_avg = self.volume_sma[-1]
        vwap_now = self.vwap[-1]
        vwap_prev = self.vwap[-2] if len(self.vwap) > 1 else vwap_now
        
        # Long Conditions
        long_trigger = (close_price > self.bb_upper[-1] and 
                       vwap_now > vwap_prev and 
                       current_volume > 1.5 * volume_avg)
        
        # Short Conditions
        short_trigger = (close_price < self.bb_lower[-1] and 
                        vwap_now < vwap_prev and 
                        current_volume > 1.5 * volume_avg)
        
        open_trades = [t for t in self.trades if t.is_open]
        
        # 🌙 Position Sizing & Entry
        if not block_entries and len(open_trades) < self.max_concurrent_trades:
            if long_trigger and not self.position.is_long:
                entry_price = self.data.Open[-1]
                sl_price = self.bb_lower[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - sl_price
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=sl_price)
                        print(f"🌕 MOON DEV LONG ENTRY: {entry_price:.2f} | Size: {size} | SL: {sl_price:.2f}")
            
            if short_trigger and not self.position.is_short:
                entry_price = self.data.Open[-1]
                sl_price = self.bb_upper[-1]
                risk_amount = self.e