Here's the complete fixed code with Moon Dev themed improvements, addressing all technical issues while preserving the original strategy logic:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Data Handling - Lunar Precision Loading 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare columns with Cosmic Care ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VortexChaikinFlow(Strategy):
    cmf_period = 20
    kc_ema_period = 20
    kc_atr_period = 20
    kc_multiplier = 2
    swing_period = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Vortex Indicator - Cosmic Trend Detection 🌌
        self.vi_plus, self.vi_minus = self.I(talib.VORTEX, 
                                            self.data.High, 
                                            self.data.Low, 
                                            self.data.Close, 
                                            timeperiod=14, 
                                            name=['VI+', 'VI-'])
        
        # Chaikin Money Flow - Lunar Money Tide Analysis 🌊
        self.cmf = self.I(talib.CMF, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.data.Volume,
                         timeperiod=self.cmf_period,
                         name='CMF')
        
        # Keltner Channel - Orbital Price Boundaries 🛰️
        self.ema20 = self.I(talib.EMA, self.data.Close, self.kc_ema_period, name='EMA20')
        self.atr20 = self.I(talib.ATR, 
                           self.data.High, 
                           self.data.Low, 
                           self.data.Close,
                           self.kc_atr_period,
                           name='ATR20')
        self.upper_kc = self.I(lambda: self.ema20 + self.kc_multiplier * self.atr20,
                              name='Upper KC')
        self.lower_kc = self.I(lambda: self.ema20 - self.kc_multiplier * self.atr20,
                              name='Lower KC')
        
        # Swing High/Low - Lunar Terrain Mapping 🗺️
        self.swing_high = self.I(talib.MAX, 
                                self.data.High, 
                                self.swing_period,
                                name='Swing High')
        self.swing_low = self.I(talib.MIN,
                               self.data.Low,
                               self.swing_period,
                               name='Swing Low')
        
        print("🌙✨ VortexChaikinFlow Strategy Activated with Lunar Precision! 🚀")

    def next(self):
        equity = self.equity()
        price = self.data.Close[-1]
        
        # Long Entry Logic - Moonrise Signal 🌕
        if (not self.position and 
            self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1] and 
            self.cmf[-1] > 0.05):
            
            sl_price = self.swing_low[-1]
            risk_per_share = price - sl_price
            if risk_per_share > 0:
                size = int(round((equity * self.risk_pct) / risk_per_share))
                if size > 0:  # Ensure valid position size
                    self.buy(size=size, sl=sl_price, tag='🌙 Long Entry')
                    print(f"🚀 BUY SIGNAL: {size} units at {price:.2f} | SL: {sl_price:.2f} | Equity: {equity:.2f}")
        
        # Short Entry Logic - Moonset Signal 🌑
        if (not self.position and 
            self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1] and 
            self.cmf[-1] < -0.05):
            
            sl_price = self.swing_high[-1]
            risk_per_share = sl_price - price
            if risk_per_share > 0