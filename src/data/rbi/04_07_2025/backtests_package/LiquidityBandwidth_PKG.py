Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
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

class LiquidityBandwidth(Strategy):
    def init(self):
        # Initialize indicators with proper self.I() wrapping 🌙
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_mid = self.I(talib.SMA, self.data.Close, timeperiod=20, name='BB_MID')
        
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume, length=20, name='VWAP')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA')
        
        # Track BB width history for percentile calculation
        self.bb_width_history = []
        
    def next(self):
        # Moon Dev debug prints ✨
        print(f"\n🌙 MOON DEV CYCLE START 🌙 | Bar: {len(self.data)-1} | Price: {self.data.Close[-1]:.2f}")
        
        # Calculate current BB width
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_mid[-1]
        self.bb_width_history.append(bb_width)
        
        # Liquidity cluster calculation 🎯
        lookback = 20
        if len(self.data) > lookback:
            vol_window = self.data.Volume[-lookback:]
            max_vol_idx = np.argmax(vol_window)
            cluster_high = self.data.High[-lookback:][max_vol_idx]
            cluster_low = self.data.Low[-lookback:][max_vol_idx]
            print(f"🔍 Liquidity Cluster | High: {cluster_high:.2f} | Low: {cluster_low:.2f}")
            
            # Entry conditions check 🚦
            current_close = self.data.Close[-1]
            vwap_condition = current_close > self.vwap[-1] if current_close > cluster_high else current_close < self.vwap[-1]
            
            if len(self.bb_width_history) > 100:
                bb_percentile = np.sum(np.array(self.bb_width_history[-100:]) < bb_width)/100
                
                if (bb_percentile <= 0.2 and 
                    self.data.Volume[-1] > 1.5*self.vol_sma[-1] and
                    vwap_condition):
                    
                    # Risk management calculations 🛡️
                    atr = self.atr[-1]
                    if current_close > cluster_high:
                        stop_loss = min(cluster_low, current_close - 0.75*atr)
                        direction = 'LONG'
                    else:
                        stop_loss = max(cluster_high, current_close + 0.75*atr)
                        direction = 'SHORT'
                        
                    risk_amount = self.equity * 0.01
                    risk_per_share = abs(current_close - stop_loss)
                    position_size = int(round(risk_amount/risk_per_share))
                    
                    # Volatility adjustment 🌪️
                    if bb_width > np.percentile(self.bb_width_history, 50):
                        position_size = position_size//2
                        print("⚡ Volatility Reduction Activated!")