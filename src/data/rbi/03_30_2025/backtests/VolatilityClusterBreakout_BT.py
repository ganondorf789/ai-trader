```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np
import pandas_ta as ta

# Data handling
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityClusterBreakout(Strategy):
    chaikin_period = 10
    keltner_ema = 20
    atr_period = 14
    risk_pct = 0.01
    consecutive_losses = 0
    
    def init(self):
        # Chaikin Volatility components
        self.ema_high = self.I(talib.EMA, self.data.High, self.chaikin_period)
        self.ema_low = self.I(talib.EMA, self.data.Low, self.chaikin_period)
        
        # Keltner Channel
        self.ema20 = self.I(talib.EMA, self.data.Close, self.keltner_ema)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        self.chaikin_values = []

    def next(self):
        if len(self.data) < 100:
            return
            
        # Calculate Chaikin Volatility
        chaikin = ((self.ema_high[-1] - self.ema_low[-1]) / self.ema_low[-1]) * 100
        self.chaikin_values.append(chaikin)
        if len(self.chaikin_values) > 100:
            self.chaikin_values.pop(0)
            
        # Keltner Bands
        upper_band = self.ema20[-1] + 2 * self.atr14[-1]
        lower_band = self.ema20[-1] - 2 * self.atr14[-1]
        
        # Entry conditions
        if not self.position and self.consecutive_losses < 3:
            if len(self.chaikin_values) >= 100:
                vol_threshold = np.percentile(self.chaikin_values, 20)
                
                if chaikin < vol_threshold:
                    # Long setup
                    if self.data.Close[-1] > upper_band:
                        risk_amount = self.equity * self.risk_pct
                        entry_price = self.data.Open[-1]
                        stop_loss = max(self.swing_low[-1], entry_price - 1.5*self.atr14[-1])
                        size = int(round(risk_amount / (entry_price - stop_loss)))
                        size = min(size, int(0.1*self.equity/entry_price))
                        
                        if size > 0:
                            print(f"🌙 LONG ENTRY at {entry_price:.2f} | Size: {size} ✨")
                            self.buy(size=size, sl=stop_loss)
                            
                    # Short setup        
                    elif self.data.Close[-1] < lower_band:
                        risk_amount = self.equity * self.risk_pct
                        entry_price = self.data.Open[-1]
                        stop_loss = min(self.swing_high[-1], entry_price + 1.5*self.atr14[-1])
                        size = int(round(risk_amount / (stop_loss - entry_price)))
                        size = min(size, int(0.1*self.equity/entry_price))
                        
                        if size > 0:
                            print(f"🌙 SHORT ENTRY at {entry_price:.2f} | Size: {size} ✨")
                            self.sell(size=size, sl=stop_loss)
        
        # Exit management
        for trade in self.trades:
            if trade.is_long:
                # Trailing stop
                if self.data.High[-1] >