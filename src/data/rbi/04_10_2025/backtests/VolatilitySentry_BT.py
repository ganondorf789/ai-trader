```python
# 🌙 Moon Dev Backtest Script for VolatilitySentry Strategy
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# =====================
# DATA PREPARATION
# =====================
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and prepare columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY IMPLEMENTATION
# =====================
class VolatilitySentry(Strategy):
    # Strategy parameters ✨
    bb_period = 20
    bb_dev = 2
    keltner_period = 20
    keltner_mult = 2
    pcr_ma_window = 960  # 10 days in 15m intervals
    risk_pct = 0.01
    atr_period = 20
    
    def init(self):
        # 🌗 Bollinger Bands Calculation
        def bb_upper(price): return talib.BBANDS(price, self.bb_period, self.bb_dev, self.bb_dev)[0]
        def bb_middle(price): return talib.BBANDS(price, self.bb_period, self.bb_dev, self.bb_dev)[1]
        def bb_lower(price): return talib.BBANDS(price, self.bb_period, self.bb_dev, self.bb_dev)[2]
        
        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_middle = self.I(bb_middle, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        # 📉 Bollinger Bandwidth
        self.bb_width = self.I(lambda u, l, m: (u-l)/m, 
                             self.bb_upper, self.bb_lower, self.bb_middle)
        
        # 📈 Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, self.keltner_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.keltner_upper = self.I(lambda e, a: e + self.keltner_mult*a, self.ema, self.atr)
        self.keltner_lower = self.I(lambda e, a: e - self.keltner_mult*a, self.ema, self.atr)
        
        # 🎚️ Volume Oscillator
        self.vol_osc = self.I(lambda v: talib.SMA(v, 14) - talib.SMA(v, 28), self.data.Volume)
        
        # 📊 PCR Calculations
        self.pcr = self.data['put_call_ratio']  # Must exist in source data
        self.pcr_ma = self.I(talib.SMA, self.pcr, self.pcr_ma_window)
        
    def next(self):
        # 🌙✨ Core Strategy Logic
        if len(self.data) < 17280:  # 6-month warmup
            return
            
        # Volatility contraction check
        bb_percentile = np.percentile(self.bb_width[-17280:], 20)
        volatility_contraction = self.bb_width[-1] < bb_percentile
        
        # PCR divergence calculation
        pcr_divergence = (abs(self.pcr[-1] - self.pcr_ma[-1]) / self.pcr_ma[-1]) > 0.25
        
        # Entry conditions
        if not self.position:
            # Long Entry 🌙🚀
            if (volatility_contraction and
                self.pcr[-1] > self.pcr_ma[-1] and
                crossover(self.data.Close, self.bb_upper) and
                self.vol_osc[-1] > 0):
                
                risk_amount = self.equity * self.risk_pct
                sl_price = self.keltner_lower[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                position