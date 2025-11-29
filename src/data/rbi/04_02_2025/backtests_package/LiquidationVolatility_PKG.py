Here's the fixed code with all indicators properly implemented without any backtesting.lib dependencies, along with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as pta
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Calculate VWAP
data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
data['Cumulative_TPV'] = (data['TP'] * data['Volume']).cumsum()
data['Cumulative_Volume'] = data['Volume'].cumsum()
data['VWAP'] = data['Cumulative_TPV'] / data['Cumulative_Volume']
data = data.drop(['TP', 'Cumulative_TPV', 'Cumulative_Volume'], axis=1)
data.index = pd.to_datetime(data['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]

class LiquidationVolatility(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR20')
        self.keltner_upper = self.I(lambda: self.ema20 + 2 * self.atr20, name='Keltner_Upper')
        self.keltner_lower = self.I(lambda: self.ema20 - 2 * self.atr20, name='Keltner_Lower')
        
        # Liquidation zones
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing_Low')
        
        # Volume and volatility
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')
        self.vw_vol = self.I(lambda: (self.data.High - self.data.Low) * self.data.Volume, name='VW_Vol')
        self.vw_vol_median = self.I(pta.median, self.vw_vol, length=20, name='VW_Vol_Median')
        
        # Momentum
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI14')
        self.vwap_slope = self.I(lambda: self.data.VWAP.diff(5), name='VWAP_Slope')

    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Lunar Cycle Update: Price ${price:.2f} | RSI {self.rsi[-1]:.1f} ✨")
        print(f"🌌 Celestial Indicators: EMA20={self.ema20[-1]:.2f} | ATR20={self.atr20[-1]:.2f} | VWAP Slope={self.vwap_slope[-1]:.4f}")
        
        # Entry logic
        if not self.position:
            # Long conditions
            long_trigger = (price > self.swing_high[-1] and 
                           price > self.keltner_upper[-1] and
                           self.data.Volume[-1] > self.volume_avg[-1] and
                           40 <= self.rsi[-2] <= 60 and
                           self.vwap_slope[-1] > 0)
            
            # Short conditions
            short_trigger = (price < self.swing_low[-1] and 
                            price < self.keltner_lower[-1] and
                            self.data.Volume[-1] > self.volume_avg[-1] and
                            40 <= self.rsi[-2] <=