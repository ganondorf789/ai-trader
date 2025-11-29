Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

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
        
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=20)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.lower_band = self.I(bb_lower, self.data.Close)

        # Volatility components
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20*96)  # 20-day SMA
        
        # Funding Rate analysis
        self.funding_90 = self.I(lambda s: s.rolling(2880).quantile(0.9), self.data.fundingrate)
        self.funding_75 = self.I(lambda s: s.rolling(2880).quantile(0.75), self.data.fundingrate)
        
        # Volume analysis
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20*96)  # 20-day SMA
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        
        # VW-RSI implementation
        def compute_vw_rsi(close, volume):
            delta = np.diff(close, prepend=np.nan)
            gain = np.where(delta > 0, delta*volume, 0)
            loss = np.where(delta < 0, -delta*volume, 0)
            avg_gain = talib.SMA(gain, timeperiod=14)
            avg_loss = talib.SMA(loss, timeperiod=14)
            rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        self.vw_rsi = self.I(compute_vw_rsi, self.data.Close, self.data.Volume)
        
        print("🌙 MOON DEV: Strategy initialized with cosmic precision! 🚀✨")
        print("🌙 Indicators ready for lunar market analysis:")
        print(f"🌙 - Bollinger Bands (20,2)")
        print(f"🌙 - ATR (14) with 20-day SMA")
        print(f"🌙 - Volume Weighted RSI (14)")
        print(f"🌙 - Funding Rate Quantiles (90th, 75th)")

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