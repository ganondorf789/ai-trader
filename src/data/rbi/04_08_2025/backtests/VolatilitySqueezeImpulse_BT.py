```python
# 🌙 Moon Dev's Volatility Squeeze Impulse Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# 🚀 Cosmic Data Preparation 🚀
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeImpulse(Strategy):
    def init(self):
        # 🌌 Celestial Indicators 🌌
        def bbands(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper, middle, lower
            
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(bbands, self.data.Close)
        
        # 🌗 Phase 1: Volatility Measurement
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_pct_20 = self.I(lambda x: x.rolling(100).apply(lambda x: np.percentile(x, 20)), self.bb_width)
        self.bb_width_pct_80 = self.I(lambda x: x.rolling(100).apply(lambda x: np.percentile(x, 80)), self.bb_width)
        
        # 🌓 Phase 2: Momentum Alignment
        self.ema13 = self.I(talib.EMA, self.data.Close, 13)
        self.macd_hist = self.I(lambda x: talib.MACD(x)[2], self.data.Close)
        self.elder_impulse = self.I(self.calculate_impulse)
        
        # 🌍 Planetary Protection (Risk Management)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        print("🌙✨ Moon Dev Systems Online - Ready for Quantum Trading! 🚀")

    def calculate_impulse(self):
        ema_dir = np.where(self.ema13 > self.ema13.shift(1), 1, -1)
        macd_dir = np.where(self.macd_hist > 0, 1, -1)
        return np.where((ema_dir == 1) & (macd_dir == 1), 1, 
                      np.where((ema_dir == -1) & (macd_dir == -1), -1, 0))

    def next(self):
        # 🌠 Interstellar Position Sizing 🌠
        risk_percent = 0.02
        risk_amount = self.equity * risk_percent
        
        if not self.position:
            # 🛸 Long Entry: Cosmic Alignment Detected 🛸
            if (self.bb_width[-1] < self.bb_width_pct_20[-1] and
                self.elder_impulse[-1] == 1 and
                self.data.Close[-1] > self.bb_upper[-1] and
                self.elder_impulse[-2] != 1):
                
                atr = self.atr[-1] or 1
                position_size = int(round(risk_amount / (2 * atr)))
                if position_size > 0:
                    self.buy(size=position_size, sl=self.data.Close[-1]-2*atr)
                    print(f"🌙🚀 STARGAZER LONG! Size: {position_size} | Entry: {self.data.Close[-1]:.2f} 🌌")
            
            # 🌑 Short Entry: Dark Matter Momentum 🌑
            elif (self.bb_width[-1] < self.bb_width_pct_20[-1] and
                  self.elder_impulse[-1] == -1 and
                  self.data.Close[-1] < self.bb_lower[-1] and
                  self.elder_impulse[-2] != -1):
                  
                atr = self.atr[-1] or 1
                position