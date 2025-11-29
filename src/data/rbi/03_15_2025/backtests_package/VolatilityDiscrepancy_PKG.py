Here's the fixed and completed code with proper Moon Dev themed debug prints and ensuring no `backtesting.lib` imports or functions are used:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityDiscrepancy(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Precompute indicators using TA-Lib
        close = self.data.Close.values.astype('float64')
        high = self.data.High.values.astype('float64')
        low = self.data.Low.values.astype('float64')
        
        # Calculate indicators
        self.rsi = self.I(talib.RSI, close, timeperiod=14)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=14)
        
        # Bollinger Bands calculation
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda: upper)
        self.lower_band = self.I(lambda: lower)
        
    def next(self):
        # Skip if not enough data
        if len(self.rsi) < 2 or len(self.atr) < 2:
            return
            
        current_rsi = self.rsi[-1]
        previous_rsi = self.rsi[-2]
        current_atr = self.atr[-1]
        current_atr_ma = self.atr_ma[-1]
        current_close = self.data.Close[-1]
        
        # Moon Dev debug monitoring 🌙
        print(f"🌙 Current RSI: {current_rsi:.1f} | ATR: {current_atr:.1f} | BB Width: {self.upper_band[-1]-self.lower_band[-1]:.1f}")
        
        if not self.position:
            # Long entry: Oversold with volatility expansion
            if (current_rsi < 30 and 
                current_atr > current_atr_ma and 
                current_close <= self.lower_band[-1]):
                
                self.enter_long()
            
            # Short entry: Overbought with volatility expansion
            elif (current_rsi > 70 and 
                  current_atr > current_atr_ma and 
                  current_close >= self.upper_band[-1]):
                
                self.enter_short()
                
        else:
            # Exit conditions for existing positions
            if self.position.is_long:
                if (current_rsi > 30 and previous_rsi <= 30) or (current_atr < current_atr_ma):
                    self.position.close()
                    print(f"🌙✨ LONG EXIT: RSI {current_rsi:.1f} | ATR MA Ratio {current_atr/current_atr_ma:.2f} ✨")
            
            elif self.position.is_short:
                if (current_rsi < 70 and previous_rsi >= 70) or (current_atr < current_atr_ma):
                    self.position.close()
                    print(f"🌙✨ SHORT EXIT: RSI {current_rsi:.1f} | ATR MA Ratio {current_atr/current_atr_ma:.2f} ✨")

    def enter_long(self):
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        risk_amount = self.equity * self.risk_percent
        position_size = int(round(risk_amount / current_atr))
        
        if position_size > 0:
            sl = current_close - current_atr
            tp = current_close + 1.5 * current_atr
            self.buy(size=position_size, sl=sl, tp=tp)
            print(f"🚀🌙 LONG ENTRY! Size: {position_size} | SL: {sl:.1f