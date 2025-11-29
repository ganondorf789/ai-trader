```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR BANDWIDTHSURGE STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# ===================
# DATA PREPARATION 🌙
# ===================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'])

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py format 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

# ======================
# STRATEGY CLASS 🚀✨
# ======================
class BandwidthSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # ======================
        # INDICATORS 🌙📈
        # ======================
        def bb_upper(close):
            u, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return u
        def bb_middle(close):
            _, m, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return m
        def bb_lower(close):
            _, _, l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return l
            
        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_middle = self.I(bb_middle, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        # Bandwidth calculation 🌙
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, 
                              self.bb_upper, self.bb_lower, self.bb_middle)
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, 20)
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, 20)
        
        # ADX system 🌙
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 50)
        
        # Swing extremes 🌙
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("🌙✨ Strategy initialized! Lunar indicators ready! 🚀")

    def next(self):
        # ======================
        # ENTRY LOGIC 🌙🚀
        # ======================
        if not self.position:
            # Bandwidth contraction check
            if self.bandwidth[-1] == self.bandwidth_low[-1]:
                # Trend strength confirmation
                if (self.adx[-1] > 25 and 
                    self.adx[-1] > self.adx[-2] and 
                    self.data.Volume[-1] > self.volume_ma[-1]):
                    
                    # Directional breakout 🌙
                    price = self.data.Close[-1]
                    if price > self.bb_upper[-1]:
                        # Long setup 🌙
                        sl_price = self.swing_low[-1]
                        risk_amount = self.equity * self.risk_per_trade
                        position_size = int(round(risk_amount / (price - sl_price)))
                        
                        if position_size > 0:
                            self.buy(size=position_size, sl=sl_price)
                            print(f"🚀🌙 LONG! Size: {position_size} | Entry: {price:.2f} | SL: {sl_price:.2f}")
                            
                    elif price < self.bb_lower[-1]:
                        # Short setup 🌙
                        sl_price = self.swing_high[-1]
                        risk_amount = self.equity * self.risk_per_t