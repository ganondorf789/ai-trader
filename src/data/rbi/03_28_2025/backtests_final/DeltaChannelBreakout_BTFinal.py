I'll analyze and fix the backtest code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import pytz
from datetime import time

# Data preprocessing 🌙 Lunar Data Loading
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data ✨ Cosmic Data Cleansing
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data.index = data.index.tz_localize('UTC')

class DeltaChannelBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    max_daily_trades = 3  # 🚀 Max 3 trades per day
    swing_period = 20  # 🌌 Swing period for highs/lows
    
    def init(self):
        # Heikin-Ashi calculations 🌙 Lunar Candles
        ha = self.I(ta.heikin_ashi, self.data.Open, self.data.High, self.data.Low, self.data.Close, name=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close'])
        
        # 2h Keltner Channel (160 periods on 15m) ✨ Cosmic Channels
        self.kc_upper, self.kc_middle, self.kc_lower = self.I(
            ta.kc, self.data.High, self.data.Low, self.data.Close, 
            length=160, scalar=2.5, mamode='EMA', 
            name=['KC_UPPER', 'KC_MIDDLE', 'KC_LOWER']
        )
        
        # Cumulative Delta Volume 🚀 Rocket Fuel
        delta = np.where(self.data.Close > self.data.Open, self.data.Volume, -self.data.Volume)
        self.cum_delta = self.I(lambda: delta.cumsum(), name='CUM_DELTA')
        
        # Swing high/low indicators 🌌 Galactic Extremes
        self.ha_high = self.I(talib.MAX, ha.HA_Close, self.swing_period)
        self.ha_low = self.I(talib.MIN, ha.HA_Close, self.swing_period)
        self.delta_high = self.I(talib.MAX, self.cum_delta, self.swing_period)
        self.delta_low = self.I(talib.MIN, self.cum_delta, self.swing_period)
        
        self.trades_today = 0
        self.current_day = None
        
    def calculate_position(self, stop_loss, direction):
        """🌕 Calculate position size based on risk percentage"""
        risk_amount = self.equity * self.risk_pct
        if direction == 'long':
            price_diff = abs(self.data.Close[-1] - stop_loss)
        else:
            price_diff = abs(stop_loss - self.data.Close[-1])
            
        position_size = risk_amount / price_diff
        position_size = round(position_size, 8)  # ✨ Prevent floating point errors
        
        if direction == 'long':
            self.buy(size=position_size, sl=stop_loss)
        else:
            self.sell(size=position_size, sl=stop_loss)
        
        self.trades_today += 1
        print(f"🌙 Moon Dev Trade Alert: {direction.upper()} position opened | Size: {position_size:.4f} | SL: {stop_loss:.2f}")
        
    def next(self):
        # Time session check 🕒 Cosmic Clock
        ny_time = self.data.index[-1].astimezone(pytz.timezone('America/New_York')).time()
        session_active = time(8, 0) <= ny_time <= time(11, 30)
        
        # Reset daily trade counter 🌞 Solar Cycle Reset
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.trades_today = 0
            
        if self.trades_today >= self.max_daily_trades or not session_active:
            return
            
        ha_close = self.data.HA_