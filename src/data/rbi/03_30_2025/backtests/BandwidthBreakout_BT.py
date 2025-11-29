```python
# 🌙 Moon Dev's BandwidthBreakout Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib

# Data Preparation 🌐
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data columns 🌟
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandwidthBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    width_percentile = 20
    vol_ma_period = 20
    cks_lookback = 10
    atr_multiplier = 3
    ma50_period = 50
    risk_pct = 0.01
    max_duration = 288  # 3 days in 15m intervals
    
    def init(self):
        # 🌗 Core Indicators Calculation
        # Bollinger Bands with TA-Lib
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[0], 
                             self.data.Close, name='BB_upper')
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[1], 
                            self.data.Close, name='BB_middle')
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[2], 
                             self.data.Close, name='BB_lower')
        
        # Bandwidth Calculation ✨
        self.bb_width = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle, name='BB_Width')
        
        # Volume Confirmation 📊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period, name='Volume_MA')
        
        # Trend Filter 🚦
        self.ma50 = self.I(talib.SMA, self.data.Close, self.ma50_period, name='MA50')
        
        # Chande Kroll Stops 🛑
        self.hh = self.I(talib.MAX, self.data.High, self.cks_lookback, name='HH')
        self.ll = self.I(talib.MIN, self.data.Low, self.cks_lookback, name='LL')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.cks_lookback, name='ATR')
        
        # Trailing Stops Calculation 🌙
        self.cks_long = self.I(lambda: self.hh - self.atr_multiplier*self.atr, name='CKS_Long')
        self.cks_short = self.I(lambda: self.ll + self.atr_multiplier*self.atr, name='CKS_Short')

    def next(self):
        # Wait for sufficient data 🌌
        if len(self.data) < 100 or len(self.bb_width) < 100:
            return
        
        # Current Market Conditions 🔍
        price = self.data.Close[-1]
        width = self.bb_width[-1]
        width_percentile = np.percentile(self.bb_width[-100:], self.width_percentile)
        
        # Entry Conditions Check ✅
        bb_contraction = width < width_percentile
        vol_spike = self.data.Volume[-1] > self.vol_ma[-1]
        trend_dir = price > self.ma50[-1]
        
        # Moon Dev Entry Logic 🌙🚀
        if not self.position:
            # Long Entry 🌕
            if all([bb_contraction, vol_spike, trend_dir, price > self.bb_upper[-1]]):
                risk_amount = self.equity * self.risk_pct
                sl = self.bb_lower[-1]
                position_size = int(round(risk_amount/(price - sl)))
                
                if position_size > 0:
                    self