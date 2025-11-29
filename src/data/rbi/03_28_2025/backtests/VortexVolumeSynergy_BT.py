```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')

# DATA PREPARATION 🛠️
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexVolumeSynergy(Strategy):
    vi_period = 14
    vol_lookback = 1920  # 20 days in 15m intervals
    cmf_period = 20
    cmf_low_period = 192  # 2 days in 15m
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # VORTEX INDICATOR 🌪️
        self.vi_plus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.data.Close, self.vi_period)[0], name='VI+')
        self.vi_minus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.data.Close, self.vi_period)[1], name='VI-')
        
        # VOLUME PERCENTILE 📊
        def vol_percentile(series):
            return series.rolling(self.vol_lookback).quantile(0.9).shift(1)
        self.vol_pct = self.I(vol_percentile, self.data.Volume, name='Volume_PCT')
        
        # CHAIKIN MONEY FLOW WITH VOLUME ADJUSTMENT 💸
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.cmf_period, name='CMF')
        self.vol_sma = self.I(talib.SMA, self.data.Volume, self.cmf_period, name='Vol_SMA')
        self.adj_cmf = self.I(lambda cmf, vol, sma: cmf * (vol/sma), self.cmf, self.data.Volume, self.vol_sma, name='Adj_CMF')
        
        # CMF LOW DETECTION 🔻
        self.cmf_low = self.I(talib.MIN, self.adj_cmf, self.cmf_low_period, name='CMF_Low')
        
        # ATR FOR STOP LOSS 🛑
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
    def next(self):
        price = self.data.Close[-1]
        
        # ENTRY LOGIC 🚀
        if not self.position:
            # Vortex crossover condition
            vi_cross = crossover(self.vi_plus, self.vi_minus)
            
            # Volume spike validation
            vol_spike = self.data.Volume[-1] > self.vol_pct[-1]
            
            if vi_cross and vol_spike:
                # Calculate position size 🌙
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr[-1]
                stop_loss = price - 2*atr_value
                risk_per_share = price - stop_loss
                
                if risk_per_share <= 0:
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, tag='MoonDev_Entry')
                    print(f"🌕 MOON DEV ENTRY 🌕 | Size: {position_size} | Entry: {price:.2f} | SL: {stop_loss:.2f}")
        
        # EXIT LOGIC ✨
        else:
            # CMF 2-day low exit
            if self.adj_cmf[-1] <= self.cmf_low[-1]:
                self.position.close()
                print(f"🌑 MO