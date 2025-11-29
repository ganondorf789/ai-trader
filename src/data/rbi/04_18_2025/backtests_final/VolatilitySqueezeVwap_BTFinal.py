Here's the fixed backtest code with all technical issues resolved while preserving the original strategy logic:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLATILITY SQUEEZE VWAP STRATEGY 🚀

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Clean and prepare data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌕
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set index 🌑
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeVwap(Strategy):
    # Strategy parameters 🚀
    bb_period = 20
    bb_std = 2
    vwap_slope_period = 10
    squeeze_window = 100
    squeeze_percentile = 20
    exit_percentile = 80
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 VWAP Calculation using pandas_ta
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low, 
                          close=self.data.Close,
                          volume=self.data.Volume,
                          name='VWAP')
        
        # 🚀 Bollinger Bands Calculation
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            lambda: talib.BBANDS(self.data.Close, 
                               timeperiod=self.bb_period,
                               nbdevup=self.bb_std,
                               nbdevdn=self.bb_std,
                               matype=0),
            name='BBANDS'
        )
        
        # ✨ Bandwidth Calculation
        def calc_bb_width():
            if len(self.bb_upper) > 0 and len(self.bb_lower) > 0 and len(self.bb_mid) > 0:
                return (self.bb_upper - self.bb_lower) / self.bb_mid
            return np.zeros(len(self.data.Close))
        
        self.bb_width = self.I(calc_bb_width, name='BB_Width')
        
        # 🌕 Historical Percentile Calculation
        def pct_rank(x):
            return x.rolling(self.squeeze_window).apply(
                lambda s: (s.rank(pct=True).values[-1]*100)
            )
        
        self.bb_pct_rank = self.I(
            lambda: pct_rank(self.bb_width),
            name='BB_Pct_Rank'
        )
        
        # 🌙 VWAP Slope using linear regression
        self.vwap_slope = self.I(
            lambda: talib.LINEARREG_SLOPE(self.vwap, timeperiod=self.vwap_slope_period),
            name='VWAP_Slope'
        )
        
        # 🚀 ATR for risk management
        self.atr = self.I(
            lambda: talib.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period),
            name='ATR'
        )
        
        print("✨ MOON DEV INDICATORS INITIALIZED SUCCESSFULLY 🌙")

    def next(self):
        # Skip initial warmup period 🌑
        if len(self.data) < 100:
            return
        
        # Current values ✨
        price = self.data.Close[-1]
        vwap_slope = self.vwap_slope[-1]
        bb_pct = self.bb_pct_rank[-1]
        vwap_value = self.vwap[-1]
        atr_value = self.atr[-1]
        
        # 🌙 Moon Dev Entry Logic 🚀
        if not self.position:
            # Entry conditions
            slope_rising = vwap_slope > 0
            in_squeeze = bb_pct < self.squeeze_percentile
            price_above_vwap = price > vwap_value
            
            if all([slope_rising, in_s