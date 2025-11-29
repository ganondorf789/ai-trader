```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class MomentumBreakoutEMA(Strategy):
    ema50_period = 50
    ema200_period = 200
    adx_period = 14
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, self.ema200_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        self.signal_long = False
        self.signal_short = False
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Process pending signals from previous bar
        if self.signal_long and not self.position:
            risk_amount = self.equity * self.risk_pct
            atr_value = self.atr[-1]
            entry_price = self.data.Open[-1]
            
            if atr_value == 0:
                self.signal_long = False
                return
                
            position_size = risk_amount / (atr_value * 1)
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.buy(size=position_size)
                self.trailing_high = self.data.High[-1]
                print(f"🚀🌙 LONG ENTRY! Size: {position_size} @ {entry_price:.2f}")
            self.signal_long = False

        elif self.signal_short and not self.position:
            risk_amount = self.equity * self.risk_pct
            atr_value = self.atr[-1]
            entry_price = self.data.Open[-1]
            
            if atr_value == 0:
                self.signal_short = False
                return
                
            position_size = risk_amount / (atr_value * 1)
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.sell(size=position_size)
                self.trailing_low = self.data.Low[-1]
                print(f"🌑🌙 SHORT ENTRY! Size: {position_size} @ {entry_price:.2f}")
            self.signal_short = False

        # Detect new signals
        if crossover(self.ema50, self.ema200) and self.adx[-1] > 25:
            print(f"✨🌙 GOLDEN CROSS! EMA50({self.ema50[-1]:.2f}) > EMA200({self.ema200[-1]:.2f}) ADX: {self.adx[-1]:.2f}")
            self.signal_long = True
            
        elif crossunder(self.ema50, self.ema200) and self.adx[-1] > 25:
            print(f"💀🌙 DEATH CROSS! EMA50({self.ema50[-1]:.2f}) < EMA200({self.ema200[-1]:.2f}) ADX: {self.adx[-1]:.2f}")
            self.signal_short = True

        # Manage exits
        if self.position.is_long:
            self.trailing_high = max(self.trailing_high, self.data.High[-1])
            stop_level = self.trailing_high - 2 * self.atr[-1]
            
            if self.data.Low[-1] <= stop_level:
                self.position.close()
                print(f"🌙🔻 LONG EXIT! Trail High: {self.trailing_high:.2f} Stop: {stop_level:.2f}")

        elif self.position.is_short:
            self.trailing_low = min(self.trailing_low, self.data.Low[-1])
            stop_level = self.trailing_low + 2 * self.atr[-1]
            
            if self.data.High[-1] >= stop_level:
                self.position.close()
                print(f"🌙🔺 SHORT EXIT! Trail Low: {self.trailing_low:.2f} Stop: {stop_level:.2f}")

# Data preprocessing
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-t