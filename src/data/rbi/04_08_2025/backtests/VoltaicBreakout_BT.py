Here's the complete backtesting implementation for the VoltaicBreakout strategy:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VoltaicBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])
        
        # Calculate indicators using TA-Lib and pandas_ta
        self.choppiness = self.I(self.calculate_choppiness)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.consolidation_high = self.I(talib.MAX, self.data.High, 20)
        self.consolidation_low = self.I(talib.MIN, self.data.Low, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def calculate_choppiness(self):
        return ta.chop(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=14
        )
    
    def next(self):
        if self.position:
            return  # Exit if already in position
            
        # Common conditions
        ci_below_threshold = self.choppiness[-1] < 38.2
        current_volume = self.data.Volume[-1]
        vol_spike = current_volume > self.vol_sma[-1] * 1.5
        
        # Long entry conditions
        price_breakout_long = self.data.Close[-1] > self.consolidation_high[-1]
        rsi_bull_div = (self.data.Low[-1] < self.data.Low[-2] and 
                        self.rsi[-1] > self.rsi[-2])
        
        # Short entry conditions
        price_breakout_short = self.data.Close[-1] < self.consolidation_low[-1]
        rsi_bear_div = (self.data.High[-1] > self.data.High[-2] and 
                         self.rsi[-1] < self.rsi[-2])
        
        # Execute trades
        if ci_below_threshold and vol_spike:
            if price_breakout_long and rsi_bull_div:
                self.enter_long()
                
            elif price_breakout_short and rsi_bear_div:
                self.enter_short()
    
    def enter_long(self):
        entry_price = self.data.Close[-1]
        stop_loss = self.data.Low[-1]  # Breakout candle low
        atr_stop = entry_price - 1.5 * self.atr[-1]
        stop_loss = max(stop_loss, atr_stop)  # Use tighter stop
        
        risk = entry_price - stop_loss
        position_size = self.calculate_size(risk)
        
        if position_size > 0:
            take_profit = entry_price + (self.consolidation_high[-1] - 
                                       self.consolidation_low[-1]) * 1.618
            self.buy(
                size=int(round(position_size)),
                sl=stop_loss,
                tp=take_profit
            )
            print(f"🌙✨🚀 MOON DEV LONG ENTRY! Price: {entry_price:.2f}, "
                  f"Size: {position_size:.0f}, TP: {take_profit:.2f}, "
                  f"SL: {stop_loss:.2f}")
    
    def enter_short(self):
        entry_price = self.data.Close[-1]
        stop_loss = self.data.High[-1]  # Breakout candle high
        atr_stop = entry_price + 1.5 * self.atr[-1]
        stop_loss = min(stop_loss, atr_stop)  # Use tighter stop
        
        risk = stop_loss - entry_price
        position_size = self.calculate_size(risk)
        
        if position_size > 0:
            take_profit = entry_price - (self.consolidation_high[-1] - 
                                        self.consolidation_low[-1]) * 1.618
            self.sell(
                size=int(round(position_size)),
                sl=stop_loss,
                tp=take_pro