Here's the fixed and completed backtest code with all Moon Dev debug enhancements and proper position sizing implementation:

```python
# 🌙 Moon Dev's VolClusterSqueeze Backtest Implementation
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# 🚀 Data Preparation Rocket Launch!
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

class VolClusterSqueeze(Strategy):
    risk_pct = 0.01  # 🌕 1% Risk Per Trade
    
    def init(self):
        # 🌗 Bollinger Bands Configuration
        self.upper, self.mid, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                 timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 🌌 Volatility Squeeze Detection
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, self.upper, self.lower, self.mid)
        self.max_bw = self.I(talib.MAX, self.bandwidth, 100)
        self.min_bw = self.I(talib.MIN, self.bandwidth, 100)
        self.bw_percentile = self.I(lambda b, mn, mx: (b - mn)/(mx - mn + 1e-9), 
                                  self.bandwidth, self.min_bw, self.max_bw)
        
        # 💧 VWAP Liquidity Pools
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        
        # 📈 Volume Confirmation System
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # ✨ Moon Dev Debug Trackers
        self.last_squeeze = 0
        self.trade_count = 0

    def next(self):
        # 🌙 Lunar Cycle Debug Prints
        if len(self.data) % 500 == 0:
            print(f"🌒 Moon Phase Update | Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | Equity: {self.equity:,.2f}")
        
        # 🛑 Existing Position Management
        if self.position:
            pos = self.position
            entry = pos.entry_price
            
            # 🎯 Dynamic Exit Conditions
            if pos.is_long:
                sl_level = min(self.mid[-1], self.vwap[-1])
                tp_level = self.upper[-1] + (self.upper[-1] - self.lower[-1])
                
                if self.data.Close[-1] <= sl_level:
                    print(f"🌧️ Cloud Cover SL | LONG Exit {self.data.Close[-1]:.2f} | PnL: {pos.pl:.2f}")
                    pos.close()
                elif self.data.Close[-1] >= tp_level:
                    print(f"🚀 Galactic TP | LONG Profit {self.data.Close[-1]:.2f} | PnL: {pos.pl:.2f}")
                    pos.close()
                    
            elif pos.is_short:
                sl_level = max(self.mid[-1], self.vwap[-1])
                tp_level = self.lower[-1] - (self.upper[-1] - self.lower[-1])
                
                if self.data.Close[-1] >= sl_level:
                    print(f"🌧️ Meteor Shower SL | SHORT Exit {self.data.Close[-1]:.2f} | PnL: {pos.pl:.2f}")
                    pos.close()
                elif self.data.Close[-1] <= tp_level:
                    print(f"🚀 Cosmic TP | SHORT Profit {self.data.Close[-1]:.2f} | PnL: {pos.pl:.2f}")
                    pos.close()
        
        # 🚦 Entry Signal Conditions
        else:
            # 🔄 Mean Reversion Squeeze
            squeeze_active = self.bw_percentile[-1] < 0.1
            vwap_cluster = abs(self.data.Close[-1] - self.vwap[-1])/self.vwap[-1] < 0.005
            vol_boost =