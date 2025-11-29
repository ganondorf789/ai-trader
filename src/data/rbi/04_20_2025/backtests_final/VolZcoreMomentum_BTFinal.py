```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib as ta
import numpy as np

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolZcoreMomentum(Strategy):
    def init(self):
        # Volume indicators
        self.volume_sma20 = self.I(ta.SMA, self.data.Volume, timeperiod=20)
        
        # Volatility-adjusted returns calculation
        def var_calc(close, volume, sma_period):
            rocp = ta.ROCP(close, timeperiod=1)
            sma_vol = ta.SMA(volume, timeperiod=sma_period)
            return rocp * volume / sma_vol
        self.var = self.I(var_calc, self.data.Close, self.data.Volume, 20)
        
        # Z-score calculation
        def zscore_calc(var, period):
            mean = ta.SMA(var, timeperiod=period)
            std = ta.STDDEV(var, timeperiod=period)
            return (var - mean) / std
        self.z_score = self.I(zscore_calc, self.var, 20)
        
        # Percentile calculations
        def percentile(series, period, q):
            return series.rolling(period).quantile(q)
        self.percentile_75 = self.I(percentile, self.z_score, 100, 0.75)
        self.percentile_25 = self.I(percentile, self.z_score, 100, 0.25)
        self.percentile_50 = self.I(percentile, self.z_score, 100, 0.5)
        self.percentile_90 = self.I(percentile, self.z_score, 100, 0.9)
        self.percentile_10 = self.I(percentile, self.z_score, 100, 0.1)
        
        # Volatility indicator
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙 Lunar Launch Initiated! Systems Go! 🚀")

    def next(self):
        if len(self.data) < 100:
            return

        current_z = self.z_score[-1]
        current_volume = self.data.Volume[-1]
        vol_sma = self.volume_sma20[-1]
        atr = self.atr[-1]
        price = self.data.Close[-1]
        equity = self.equity

        # Risk calculations
        risk_amount = 0.01 * equity
        max_size = 0.05 * equity / price
        stop_loss = price - (2 * atr) if current_z > self.percentile_75[-1] else price + (2 * atr)
        position_size = int(round(min(risk_amount / (2 * atr * price), max_size)))

        if not position_size:
            return

        # Trade logic
        if not self.position:
            if current_z > self.percentile_75[-1] and current_volume > vol_sma:
                self.buy(size=position_size, sl=stop_loss, tag="🌕 MOONSHOT LONG")
                print(f"🚀 LIFTOFF! Long {position_size} units @ {price:.2f} | SL: {stop_loss:.2f}")
                
            elif current_z < self.percentile_25[-1] and current_volume > vol_sma:
                self.sell(size=position_size, sl=stop_loss, tag="🌑 DARK SIDE SHORT")
                print(f"🌌 BLACK HOLE! Short {position_size} units @ {price:.2f} | SL: {stop_loss:.2f}")
                
        else:
            if self.position.is_long:
                if current_z < self.percentile_50[-1] or current_z > self.percentile_90[-1]:
                    self.position.close()
                    print(f"🌗 PARTIAL ECLIPSE: Closing long @ {price:.2f}")
                    
            elif self.position.is_short:
                if current_z > self.percentile_50[-1] or current_z < self.percentile_10