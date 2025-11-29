Here's the complete fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np  # Added for array operations

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

class VolatilityShiftBand(Strategy):
    def init(self):
        # Volatility indicators
        self.hv_10 = self.I(talib.STDDEV, self.data.Close, timeperiod=10, name='HV_10')
        self.median_band = self.I(talib.MEDIAN, self.hv_10, timeperiod=30, name='Median_Band')
        
        # Volume indicators
        self.volume_ma5 = self.I(talib.SMA, self.data.Volume, 5, name='Vol_MA5')
        self.volume_ma20 = self.I(talib.SMA, self.data.Volume, 20, name='Vol_MA20')
        
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, 20, 2, 2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, 20, 2, 2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_Lower')
        
        # Risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        self.trade_risk_pct = 0.02
        self.entry_price = None
        self.stop_loss = None

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 40:
            return
        
        # Current indicator values
        hv = self.hv_10[-1]
        med = self.median_band[-1]
        vol_ok = self.volume_ma5[-1] > self.volume_ma20[-1]
        atr = self.atr[-1]
        
        # Calculate volatility ratio
        if hv > med:
            vol_ratio = (hv - med) / med
        else:
            vol_ratio = (med - hv) / hv

        # Entry conditions
        if not self.position:
            # Long entry logic 🌙🚀
            if (self.hv_10[-2] < self.median_band[-2] and self.hv_10[-1] > self.median_band[-1] and 
                vol_ok and vol_ratio > 0.15):
                self.entry_price = self.data.Close[-1]
                risk_amount = self.equity * self.trade_risk_pct
                risk_per_unit = self.entry_price - (self.entry_price - 2*atr)
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    self.buy(size=size)
                    self.stop_loss = self.entry_price - 2*atr
                    print(f"🌙🚀 MOONSHOT ALERT! Long {size} units @ {self.entry_price:.2f} | SL: {self.stop_loss:.2f}")
                    print(f"🌕 Volatility Ratio: {vol_ratio:.2%} | Volume Confirmed: {vol_ok}")
            
            # Short entry logic 🌙🌑
            elif (self.median_band[-2] < self.hv_10[-2] and self.median_band[-1] > self.hv_10[-1] and 
                  vol_ok and vol_ratio > 0.15):
                self.entry_price = self.data.Close[-1]
                risk_amount = self.equity * self.trade_risk_pct
                risk_per_unit = (self.entry_price +