```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VoltaDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Calculate MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD,
            self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Swing high/low indicators
        self.price_swing_high = self.I(talib.MAX, self.data.High, 5)
        self.price_swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.macd_swing_high = self.I(talib.MAX, self.macd, 5)
        self.macd_swing_low = self.I(talib.MIN, self.macd, 5)
        
    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Bot scanning bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} ✨")
        
        if not self.position:
            # Long entry logic
            if (self.data.Low[-1] <= self.bb_lower[-1] and
                self._bullish_divergence()):
                
                sl_price = self.bb_lower[-1] * 0.995
                self._enter_trade('long', sl_price)
            
            # Short entry logic
            elif (self.data.High[-1] >= self.bb_upper[-1] and
                self._bearish_divergence()):
                
                sl_price = self.bb_upper[-1] * 1.005
                self._enter_trade('short', sl_price)
        else:
            # Exit logic
            if (self.position.is_long and 
                self.data.Close[-1] >= self.bb_middle[-1]):
                
                print(f"🎯 Moon Exit LONG at MA {self.bb_middle[-1]:.2f} | P&L: {self.position.pl:.2f} 💰")
                self.position.close()
                
            elif (self.position.is_short and 
                self.data.Close[-1] <= self.bb_middle[-1]):
                
                print(f"🎯 Moon Exit SHORT at MA {self.bb_middle[-1]:.2f} | P&L: {self.position.pl:.2f} 💰")
                self.position.close()
    
    def _bullish_divergence(self):
        if len(self.price_swing_low) < 6: return False
        price_lower = self.price_swing_low[-1] < self.price_swing_low[-6]
        macd_higher = self.macd_swing_low[-1] > self.macd_swing_low[-6]
        return price_lower and macd_higher
    
    def _bearish_divergence(self):
        if len(self.price_swing_high) < 6: return False
        price_higher = self.price_swing_high[-1] > self.price_swing_high[-6]
        macd_lower = self.macd_swing_high[-1] < self.macd_swing_high[-6]
        return price_higher and macd_lower
    
    def _enter_trade(self, direction, sl_price):
        entry_price = self.data.Close[-1]
        risk_per_share = (abs(entry_price - sl_price) 
            if direction == 'long' else abs(sl_price - entry_price))
        
        if risk_per_share <= 0:
            print("🌑 Invalid risk calculation - skipping trade")
            return
        
        position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
        
        if position_size > 0:
            if direction == 'long':
                self.buy(size=position_size, sl=sl_price)
                print(f"🚀🌙 LONG Activated! Size: {position_size} | Entry: {entry_price:.2f} | SL: {sl_price:.2f}")
            else:
                self.sell(size=position