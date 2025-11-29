Here's the debugged version of your code with Moon Dev themed debug prints and technical fixes applied:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class DivergentPulse(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators
        self.ema10 = self.I(talib.EMA, self.data.Close, timeperiod=10)
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Volume analysis
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Swing detection
        self.price_lows = self.I(talib.MIN, self.data.Low, timeperiod=5)
        self.rsi_lows = self.I(talib.MIN, self.rsi, timeperiod=5)
        
        # Trade tracking
        self.trailing_stop = None
        self.peak_price = None

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Entry conditions
            ema_cross = (self.ema10[-2] < self.sma50[-2] and self.ema10[-1] > self.sma50[-1])
            rsi_overbought = self.rsi[-1] > 70
            volume_spike = self.data.Volume[-1] > 1.5 * self.vol_sma[-1]
            
            # Divergence check
            if len(self.price_lows) > 2 and len(self.rsi_lows) > 2:
                price_divergence = (self.price_lows[-1] < self.price_lows[-2] and
                                  self.rsi_lows[-1] > self.rsi_lows[-2])
            else:
                price_divergence = False
                
            if ema_cross and rsi_overbought and price_divergence and volume_spike:
                atr_value = self.atr[-1]
                stop_loss = price - 1.5 * atr_value
                risk_amount = self.risk_percent * self.equity
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.trailing_stop = stop_loss
                        self.peak_price = price
                        print(f"🌙✨🚀 Lunar Boost: LONG {position_size} @ {price:.2f} | SL: {stop_loss:.2f} ✨")
        else:
            # Update trailing stop
            self.peak_price = max(self.peak_price, self.data.High[-1])
            new_stop = self.peak_price - 1.5 * self.atr[-1]
            self.trailing_stop = max(self.trailing_stop, new_stop)
            
            # Exit conditions
            if self.data.Low[-1] <= self.trailing_stop:
                self.position.close()
                print(f"🌙💥 Cosmic Stop: Exited @ {self.data.Close[-1]:.2f} | P/L: {self.position.pl:.2f}")
            elif (self.sma50[-2] > self.ema10[-2] and self.sma50[-1] < self.ema10[-1]):
                self.position.close()
                print(f"🌙📉 Gravity Pull: EMA/SMA Cross @ {self.data.Close[-1]:.2f}")

# Backtest setup
bt = Backtest(data, DivergentPulse, cash=10000