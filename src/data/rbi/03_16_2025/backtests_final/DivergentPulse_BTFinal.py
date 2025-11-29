I'll help fix the code while maintaining the strategy logic. Here's the corrected version with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and clean data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class DivergentPulse(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Indicators with self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.macd, self.macd_signal, self.macd_hist = self.I(self._compute_macd)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(self._compute_bollinger_bands)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=5)
        self.price_lows = self.I(talib.MIN, self.data.Low, timeperiod=5)
        self.macd_lows = self.I(talib.MIN, self.macd_hist, timeperiod=5)
        
        self.trade_count = 0
        self.current_date = None
        self.has_rsi_risen = False

    def _compute_macd(self):
        return talib.MACD(self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    def _compute_bollinger_bands(self):
        return talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    def next(self):
        current_date = self.data.index[-1].date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.trade_count = 0
            print(f"🌕 New day detected! {current_date.strftime('%Y-%m-%d')} | Trades reset")

        if self.position:
            # Exit conditions
            current_high = self.data.High[-1]
            entry_price = self.position.entry_price
            
            # Upper BB touch
            if current_high >= self.upper_bb[-1]:
                print(f"🚀 UPPER BB TOUCH | Price: {current_high:.2f}")
                self.position.close()
                return
            
            # 2% profit target
            if current_high >= entry_price * 1.02:
                print(f"🎯 PROFIT TARGET HIT | 2% gain")
                self.position.close()
                return
            
            # Emergency RSI exit
            if self.has_rsi_risen and self.rsi[-1] < 30:
                print(f"🌧️ RSI DROP EMERGENCY | Current RSI: {self.rsi[-1]:.1f}")
                self.position.close()
                return
            
            # Track RSI movement
            if self.rsi[-1] > 30:
                self.has_rsi_risen = True

        else:
            # Time filter (first 2 hours)
            hour = self.data.index[-1].hour
            if hour >= 2:
                return
            
            # Entry conditions
            if (self.trade_count >= 3 or
                self.rsi[-1] >= 30 or
                self.volume_sma[-1] is None):
                return

            # Divergence check
            price_divergence = (self.price_lows[-1] < self.price_lows[-2] and
                              self.macd_lows[-1] > self.macd_lows[-2])
            
            # Volume confirmation
            volume_ok = self.data.Volume[-1] > self.volume_sma[-1]

            if price_divergence and volume_ok:
                # Risk management
                atr_value = self.atr[-1]
                stop_loss = self.data.Close