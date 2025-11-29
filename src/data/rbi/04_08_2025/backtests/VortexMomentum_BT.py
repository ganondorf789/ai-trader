```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as pta
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators with proper self.I() wrapping
        high, low, close = self.data.High, self.data.Low, self.data.Close
        
        # Vortex Indicator using pandas_ta
        vortex = pta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda: vortex['VTX_14+'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VTX_14-'], name='VI-')
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, close, timeperiod=14, name='CMO')
        
        # Volume MA and ATR
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Vol MA')
        self.atr = self.I(talib.ATR, high, low, close, 14, name='ATR')
        
        print("🌙✨ VortexMomentum Strategy Activated! Moon Dev Power Engaged! 🚀")
    
    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Entry logic
        if not self.position:
            # Long entry conditions
            if (crossover(self.vi_plus, self.vi_minus) and
                self.cmo[-1] < -50 and
                current_volume > 1.5 * self.volume_ma[-1]):
                
                if self._validate_volatility(current_close):
                    self._enter_long(current_close)
            
            # Short entry conditions
            elif (crossover(self.vi_minus, self.vi_plus) and
                  self.cmo[-1] > 50 and
                  current_volume > 1.5 * self.volume_ma[-1]):
                
                if self._validate_volatility(current_close):
                    self._enter_short(current_close)
        
        # Exit logic
        else:
            if self.position.is_long:
                self._update_long_trailing_stop()
                if crossover(self.vi_minus, self.vi_plus):
                    self.position.close()
                    print(f"🌙⚡ VI Reversal! Closing LONG at {current_close} ⚠️")
                    
            elif self.position.is_short:
                self._update_short_trailing_stop()
                if crossover(self.vi_plus, self.vi_minus):
                    self.position.close()
                    print(f"🌙⚡ VI Reversal! Closing SHORT at {current_close} ⚠️")

    def _validate_volatility(self, price):
        return self.atr[-1] >= 0.005 * price

    def _enter_long(self, price):
        atr_value = self.atr[-1]
        stop_loss = price - 1.5 * atr_value
        position_size = self._calculate_position_size(price, stop_loss)
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_loss)
            self.highest_high = self.data.High[-1]
            print(f"🌙🚀 LONG ENTRY! Size: {position_size} @ {price} | SL: {stop_loss} 🌕")

    def _enter_short(self, price):
        atr_value = self.atr[-1]
        stop_loss = price + 1.5 * atr_value
        position_size = self._calculate_position_size(price, stop_loss, short=True)
        
        if position_size > 0:
            self.sell(size=position