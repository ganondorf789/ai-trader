```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolatilitySurgeReversion(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    
    def init(self):
        # Clean data and handle columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        self.bb_middle = self.I(talib.SMA, self.data.Close, 20)
        self.bb_stddev = self.I(talib.STDDEV, self.data.Close, 20)
        self.bb_upper = self.I(lambda: self.bb_middle + 3*self.bb_stddev)
        self.bb_lower = self.I(lambda: self.bb_middle - 3*self.bb_stddev)
        
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Track extremes for trailing stops
        self.extreme_high = None
        self.extreme_low = None

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        print(f"🌙 Processing {self.data.index[-1]} | Close: {current_close} | Vol: {current_volume} ✨")
        
        if not self.position:
            # Long entry condition
            if current_close < self.bb_lower[-1] and current_volume >= 3*self.volume_ma[-1]:
                atr_val = self.atr[-1]
                stop_distance = self.atr_multiplier * atr_val
                position_size = int(round((self.equity * self.risk_percent) / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.extreme_high = self.data.High[-1]
                    print(f"🚀🌙 LONG ENTRY! Size: {position_size} | Entry: {self.data.Open[-1]} | ATR: {atr_val:.2f}")
            
            # Short entry condition
            elif current_close > self.bb_upper[-1] and current_volume >= 3*self.volume_ma[-1]:
                atr_val = self.atr[-1]
                stop_distance = self.atr_multiplier * atr_val
                position_size = int(round((self.equity * self.risk_percent) / stop_distance))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    self.extreme_low = self.data.Low[-1]
                    print(f"🚀🌙 SHORT ENTRY! Size: {position_size} | Entry: {self.data.Open[-1]} | ATR: {atr_val:.2f}")
        
        else:
            if self.position.is_long:
                # Update trailing stop
                self.extreme_high = max(self.extreme_high, self.data.High[-1])
                trailing_stop = self.extreme_high - self.atr_multiplier*self.atr[-1]
                
                # Check exit conditions
                if crossover(self.data.Close, self.bb_middle):
                    self.position.close()
                    print(f"🎯🌙 LONG EXIT (SMA) | Price: {current_close}")
                elif self.data.Low[-1] <= trailing_stop:
                    self.position.close()
                    print(f"🛑🌙 LONG EXIT (Trail) | Price: {current_close}")
            
            elif self.position.is_short:
                # Update trailing stop
                self.extreme_low = min(self.extreme_low, self.data.Low[-1])
                trailing_stop = self.extreme_low + self.atr_multiplier*self.atr[-1]
                
                # Check exit conditions
                if crossunder(self.data.Close, self.bb_middle):
                    self.position.close()
                    print(f"🎯🌙 SHORT EXIT (SMA) | Price: {current_close}")
                elif self.data.High[-1] >= trailing_stop:
                    self.position.close()
                    print(f"🛑🌙 SHORT EXIT (Trail) | Price: {current_close}")

# Data preprocessing
data = pd.read_csv('/Users/md