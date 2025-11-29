```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import pytz
from datetime import time

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data
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
    risk_pct = 0.01
    max_daily_trades = 3
    swing_period = 20
    
    def init(self):
        # Heikin-Ashi calculations 🌙
        ha = self.I(ta.heikin_ashi, self.data.Open, self.data.High, self.data.Low, self.data.Close, name=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close'])
        
        # 2h Keltner Channel (160 periods on 15m) ✨
        self.kc_upper, self.kc_middle, self.kc_lower = self.I(
            ta.kc, self.data.High, self.data.Low, self.data.Close, 
            length=160, scalar=2.5, mamode='EMA', 
            name=['KC_UPPER', 'KC_MIDDLE', 'KC_LOWER']
        )
        
        # Cumulative Delta Volume 🚀
        delta = np.where(self.data.Close > self.data.Open, self.data.Volume, -self.data.Volume)
        self.cum_delta = self.I(lambda: delta.cumsum(), name='CUM_DELTA')
        
        # Swing high/low indicators 🌌
        self.ha_high = self.I(talib.MAX, ha.HA_Close, self.swing_period)
        self.ha_low = self.I(talib.MIN, ha.HA_Close, self.swing_period)
        self.delta_high = self.I(talib.MAX, self.cum_delta, self.swing_period)
        self.delta_low = self.I(talib.MIN, self.cum_delta, self.swing_period)
        
        self.trades_today = 0
        self.current_day = None
        
    def next(self):
        # Time session check 🕒
        ny_time = self.data.index[-1].astimezone(pytz.timezone('America/New_York')).time()
        session_active = time(8,0) <= ny_time <= time(11,30)
        
        # Reset daily trade counter 🌞
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.trades_today = 0
            
        if self.trades_today >= self.max_daily_trades or not session_active:
            return
            
        ha_close = self.data.HA_Close[-1]
        price_high = ha_close >= self.ha_high[-1]
        price_low = ha_close <= self.ha_low[-1]
        delta_high_div = self.cum_delta[-1] < self.delta_high[-1]
        delta_low_div = self.cum_delta[-1] > self.delta_low[-1]
        
        # Long entry 🌕
        if ha_close > self.kc_upper[-1] and price_high and delta_high_div:
            sl = min(self.kc_lower[-1], self.ha_low[-1])
            self.calculate_position(sl, direction='long')
            
        # Short entry 🌑
        elif ha_close < self.kc_lower[-1] and price_low and delta_low_div:
            sl = max(self.kc_upper[-1], self.ha_high[-1])
            self.calculate_position(sl, direction='short')
            
        # Channel exit condition 🌊
        if self.position:
            if (self.position.is_long and self.data.Close[-1] < self.kc_middle[-1]) or \
               (self.position.is_short and self.data.Close[-1] > self.kc_middle[-1]):
                self.position.close()
                print(f"🌙✨ Channel exit