```python
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
from talib import EMA, ATR, MAX, MIN

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySqueezeBreakout(Strategy):
    initial_capital = 1_000_000
    risk_per_trade = 0.01
    atr_period = 20
    
    def init(self):
        # Calculate indicators
        self.ema = self.I(EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.keltner_upper = self.ema + 2.5 * self.atr
        self.keltner_lower = self.ema - 2.5 * self.atr
        self.keltner_width = self.keltner_upper - self.keltner_lower
        self.range_high = self.I(MAX, self.data.High, timeperiod=20)
        self.range_low = self.I(MIN, self.data.Low, timeperiod=20)
        
        # Volume indicators
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, timeperiod=5)
        
        # Open interest change
        self.open_interest = self.data['open_interest']
        self.oi_pct_change = self.I(lambda x: x.pct_change(), self.open_interest)
        
        # Keltner width percentile
        self.keltner_percentile = self.I(lambda x: x.rolling(50).rank(pct=True), self.keltner_width)
        
    def next(self):
        if len(self.data) < 50:  # Warmup period
            return
        
        current_keltner_percentile = self.keltner_percentile[-1]
        volume_declining = self.volume_slope[-1] < 0
        oi_surge = self.oi_pct_change[-1] >= 0.15
        
        if not self.position:
            # Calculate position size
            atr_value = self.atr[-1]
            risk_amount = self.initial_capital * self.risk_per_trade
            position_size = int(round(risk_amount / atr_value))
            
            # Long entry conditions
            if (current_keltner_percentile <= 0.3 and
                volume_declining and
                self.data.Close[-1] > self.range_high[-2] and
                oi_surge):
                
                self.buy(size=position_size, tag="🌙 MOON DEV LONG ENTRY")
                entry_price = self.data.Close[-1]
                self.stop_loss = entry_price - atr_value
                self.take_profit = entry_price + 2 * atr_value
                print(f"🚀🌕 MOON DEV LONG TRIGGERED! Entry: {entry_price}, SL: {self.stop_loss}, TP: {self.take_profit}")
            
            # Short entry conditions
            elif (current_keltner_percentile <= 0.3 and
                  volume_declining and
                  self.data.Close[-1] < self.range_low[-2] and
                  oi_surge):
                
                self.sell(size=position_size, tag="🌙 MOON DEV SHORT ENTRY")
                entry_price = self.data.Close[-1]
                self.stop_loss = entry_price + atr_value
                self.take_profit = entry_price - 2 * atr_value
                print(f"🌑💫 MOON DEV SHORT TRIGGERED! Entry: {entry_price}, SL: {self.stop_loss}, TP: {self.take_profit}")
        
        else:
            # Exit conditions
            if self.position.is_long:
                if self.data.Close[-1] >= self.take_profit:
                    self.position.close()
                    print(f"💰✨ MOON DEV TP HIT! Long closed at {self.data.Close[-1]}")
                elif self.data.Close[-1] <= self.stop_loss:
                    self.position.close()
                    print(f"