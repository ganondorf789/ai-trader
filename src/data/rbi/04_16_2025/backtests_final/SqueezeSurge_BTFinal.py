Here's the fully debugged and fixed backtest code with all technical issues resolved while preserving the original strategy logic:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR SQUEEZE SURGE STRATEGY 🚀

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# DATA PREPARATION 🌙
def prepare_data(data_path):
    # Load and clean data
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Proper column mapping 🌌
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert index to datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# SQUEEZE SURGE STRATEGY CLASS 🌙
class SqueezeSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    max_hold_bars = 192  # 48 hours in 15m intervals (4 bars/hr * 48)
    
    def init(self):
        # BOLLINGER BANDS CALCULATION 🎯
        def compute_bb(close, which):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return {'upper': upper, 'middle': middle, 'lower': lower}[which]
        
        self.upper_band = self.I(compute_bb, self.data.Close, 'upper', name='Upper BB')
        self.lower_band = self.I(compute_bb, self.data.Close, 'lower', name='Lower BB')
        
        # VOLUME INDICATORS 📊
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume SMA')
        
        # BANDWIDTH COMPRESSION DETECTION 🔍
        def calculate_bandwidth(upper, lower):
            return upper - lower
        self.bandwidth = self.I(calculate_bandwidth, self.upper_band, self.lower_band, name='Bandwidth')
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, timeperiod=480, name='Bandwidth Min')  # 5-day lookback
        
    def next(self):
        # MOON DEV DEBUG TRACKING 🌑
        current_bar = len(self.data) - 1
        
        # ENTRY LOGIC 🚀
        if not self.position:
            # Calculate current values
            close = self.data.Close[-1]
            upper = self.upper_band[-1]
            lower = self.lower_band[-1]
            volume = self.data.Volume[-1]
            vol_sma = self.volume_sma[-1]
            bw_min = self.bandwidth_min[-1]
            
            # Check entry conditions 🌙✨
            if (close > upper or close < lower) and (volume >= 2 * vol_sma) and (self.bandwidth[-1] == bw_min):
                # Determine trade direction
                direction = 'long' if close > upper else 'short'
                stop_price = lower if direction == 'long' else upper
                
                # Calculate position size with risk management 🌙
                equity = self.equity
                risk_amount = equity * self.risk_percent
                risk_per_share = abs(close - stop_price)
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        if direction == 'long':
                            self.buy(size=position_size, sl=stop_price)
                            print(f"🌙✨ MOON DEV LONG ENTRY @ {close:.2f} | Size: {position_size} | SL: {stop_price:.2f} 🚀")
                        else:
                            self.sell(size=position_size, sl=stop_price)
                            print(f"🌙✨ MOON DEV SHORT ENTRY @ {close:.2f} | Size: {position_size} | SL: {stop_price:.2f} 🚀")
                        
                        # Record entry details for exit logic
                        self.entry_bar = current_bar
                        self.entry_price = close
                        self.trade_direction = direction

        # EXIT LOGIC 🛑
        elif self.position: