Here's the complete fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class VolSurgeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    trailing_atr_multiplier = 2
    exit_bars = 5
    
    def init(self):
        # Clean column names handled in data preprocessing
        
        # Calculate volatility-adjusted VWAP
        def compute_vwap(h, l, c, v):
            typical_price = (h + l + c) / 3
            tpv = typical_price * v
            return np.cumsum(tpv) / np.cumsum(v)
        
        self.vwap = self.I(compute_vwap, 
                          self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 14)
        self.vwap_adj = self.I(lambda v, a: v + 0.5*a, self.vwap, self.atr)
        
        # Volume surge indicators
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Bollinger Bands components
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, 
                                                self.data.Close, 
                                                timeperiod=20, 
                                                nbdevup=2, nbdevdn=2, 
                                                matype=0)
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bb_width_max = self.I(talib.MAX, self.bb_width, timeperiod=10)
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=10)
        self.bb_threshold = self.I(lambda mx, mn: 0.5*(mx - mn), 
                                  self.bb_width_max, self.bb_width_min)
        
        # Additional indicators
        self.atr5 = self.I(talib.ATR, 
                          self.data.High, self.data.Low, self.data.Close, timeperiod=5)
        
        self.trade_entry_bar = 0
        self.trailing_high = 0

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Entry conditions
            vwap_break = price > self.vwap_adj[-1]
            volume_surge = self.data.Volume[-1] > 1.5 * self.volume_sma20[-1]
            candle_size = (self.data.High[-1] - self.data.Low[-1]) > self.atr5[-1]
            
            if vwap_break and volume_surge and candle_size:
                # Calculate stop loss
                swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)[-1]
                atr_stop = price - 1.5 * self.atr[-1]
                stop_price = min(swing_low, atr_stop)
                
                if stop_price >= price:
                    print(f"🌙⚠️ Invalid stop {stop_price} >= {price}. Trade aborted.")
                    return
                
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = price - stop_price
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🌙✨🚀 MOON DEV ENTRY: {position_size} @ {price} | SL {stop_price}")
                    self.trade_entry_bar = len(self.data)
                    self.trailing_high = price
        else:
            # Update trailing stop
            if self.data.High[-1] > self.trailing_high:
                self.trailing_high = self.data.High[-1]
                new_sl = self.trailing_high - self.trailing_atr_multiplier * self.atr[-1]
                self.position.sl = new_sl
                print(f"🌙🔄 Trailing SL updated to {new_sl}")
            
            # Exit conditions
            if self.bb_width[-1] < self.bb_threshold[-1]:
                self.position.close()
                print(f"🌙✨🚀 MO