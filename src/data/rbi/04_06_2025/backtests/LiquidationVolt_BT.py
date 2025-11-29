To address the problem, we'll create a backtesting script for the LiquidationVolt strategy following the user's specifications. Here's the structured solution:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationVolt(Strategy):
    def init(self):
        # Bollinger Bands components
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.upper_band = self.I(bb_upper, self.data.Close)
        
        self.middle_band = self.I(talib.SMA, self.data.Close, 20)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.lower_band = self.I(bb_lower, self.data.Close)

        # Volatility components
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_sma = self.I(talib.SMA, self.atr, 20*96)  # 20-day SMA
        
        # Funding Rate analysis
        self.funding_90 = self.I(lambda s: s.rolling(2880).quantile(0.9), self.data.fundingrate)
        self.funding_75 = self.I(lambda s: s.rolling(2880).quantile(0.75), self.data.fundingrate)
        
        # Volume analysis
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20*96)  # 20-day SMA
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        
        # VW-RSI implementation
        def compute_vw_rsi(close, volume):
            delta = np.diff(close, prepend=np.nan)
            gain = np.where(delta > 0, delta*volume, 0)
            loss = np.where(delta < 0, -delta*volume, 0)
            avg_gain = talib.SMA(gain, 14)
            avg_loss = talib.SMA(loss, 14)
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        self.vw_rsi = self.I(compute_vw_rsi, self.data.Close, self.data.Volume)
        
        print("🌙 MOON DEV: Strategy initialized! 🚀✨")

    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Current Price: {price:.2f} | VW-RSI: {self.vw_rsi[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        if not self.position:
            # Calculate dynamic Bollinger threshold
            volatility_boost = 1.5 if self.atr[-1] > self.atr_sma[-1] else 1.0
            adj_lower = self.middle_band[-1] - (self.middle_band[-1] - self.lower_band[-1]) * volatility_boost
            
            # Entry conditions
            if (price < adj_lower and
                self.data.fundingrate[-1] > self.funding_90[-1] and
                self.data.Volume[-1] > self.vol_ma[-1]):
                
                # Risk management calculations
                risk_amount = self.equity * 0.01
                stop_price = min(price + 2*self.atr[-1], self.swing_high[-1])
                risk_per_share = stop_price - price
                
                if risk_per_