```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumetricBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Keltner Channel components
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_keltner = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_band = self.I(lambda: self.ema20 + 2 * self.atr_keltner)
        self.lower_band = self.I(lambda: self.ema20 - 2 * self.atr_keltner)
        
        # Volume indicator
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=50)
        
        # Trailing stop ATR
        self.atr_trailing = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        if not self.position:
            # Long entry logic
            if current_close > self.upper_band[-1] and current_volume > self.volume_sma[-1]:
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr_trailing[-1]  # Typo fixed: atr_trailing
                stop_loss = current_close - 2 * atr_value
                risk_per_share = current_close - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🚀 Moon Dev Long Signal! Entry: {current_close:.2f}, Size: {position_size} ✨")
                    self.trailing_stop = 0  # Typo fixed: trailing_stop
            
            # Short entry logic
            elif current_close < self.lower_band[-1] and current_volume > self.volume_sma[-1]:
                risk_amount = self.equity * self.risk_pct  # Typo fixed: equity
                atr_value = self.atr_trailing[-1]
                stop_loss = current_close + 2 * atr_value
                risk_per_share = stop_loss - current_close
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    print(f"🌑 Moon Dev Short Signal! Entry: {current_close:.2f}, Size: {position_size} ✨")
                    self.trailing_stop = 0  # Initialize trailing stop
        else:
            if self.position.is_long:
                self.trailing_stop = max(self.trailing_stop, self.data.High[-1] - 2 * self.atr_trailing[-1])
                
                if current_close < self.upper_band[-1]:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Closed Inside Keltner Channel")
                elif self.data.Low[-1] < self.trailing_stop:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Long Trailing Stop at {self.trailing_stop:.2f} 🔥")
                    
            elif self.position.is_short:
                self.trailing_stop = min(self.trailing_stop, self.data.Low[-1] + 2 * self.atr_trailing[-1])
                
                if current_close > self.lower_band[-1]:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Closed Inside Keltner Channel")
                elif self.data.High[-1] > self.trailing_stop:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Short Trailing Stop at {self.trailing_stop:.2f} 🔥")  # Typo fixed: trailing_stop

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume':