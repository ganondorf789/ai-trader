```python
# 🌙 Moon Dev's CloudConvergence Backtest 🌕
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas as pd
import numpy as np

# 🛠️ DATA PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse and align cosmic data 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class CloudConvergenceStrategy(Strategy):
    risk_percent = 0.01  # 🌑 1% risk per trade
    
    def init(self):
        # 🌗 Ichimoku Components
        tenkan_high = self.I(talib.MAX, self.data.High, 9)
        tenkan_low = self.I(talib.MIN, self.data.Low, 9)
        self.tenkan = self.I(lambda x,y: (x+y)/2, tenkan_high, tenkan_low, name='Tenkan')
        
        kijun_high = self.I(talib.MAX, self.data.High, 26)
        kijun_low = self.I(talib.MIN, self.data.Low, 26)
        self.kijun = self.I(lambda x,y: (x+y)/2, kijun_high, kijun_low, name='Kijun')
        
        # 🌊 Volatility & Volume
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # 🌠 Trade State Tracking
        self.trade_entry = None
        self.trailing_sl = None
        self.peak_price = None

    def next(self):
        # 🌌 Wait for cosmic alignment (sufficient data)
        if len(self.data.Close) < 26:
            return

        # 🌕 Current Planetary Positions
        close = self.data.Close[-1]
        volume = self.data.Volume[-1]
        atr = self.atr[-1]
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]

        # 🌓 Entry Signals
        if not self.position:
            # Conversion Line Convergence
            conv_diff = abs(tenkan - kijun)
            conv_threshold = close * 0.005  # 0.5% threshold
            
            # Volume Surge Check
            volume_ok = volume > atr * 0.5
            
            if conv_diff < conv_threshold and volume_ok:
                # 🚀 Calculate Moon Launch Parameters
                risk_amount = self.equity * self.risk_percent
                sl_distance = 2 * atr
                position_size = int(round(risk_amount / sl_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=close - sl_distance)
                    self.trade_entry = close
                    self.peak_price = close
                    print(f"🚀🌙 MOON LAUNCH! Long {position_size} @ {close:.2f} ✨")
                    print(f"    🛡️ Initial SL: {close - sl_distance:.2f} 🌑")

        # 🌈 Manage Open Position
        else:
            # Update Lunar Highs
            self.peak_price = max(self.peak_price, self.data.High[-1])
            
            # 🛡️ Dynamic Trailing Shield
            new_sl = self.peak_price - 1.5 * atr
            if new_sl > self.position.sl:
                self.position.sl = new_sl
                print(f"✨🛡️ Shields Up! Trail SL ➡️ {new_sl:.2f}")
            
            # 🪐 Profit Targets
            tp1 = self.trade_entry + atr
            tp3 = self.trade_entry + 3*atr
            
            # 🎯 Take Partial Profits
            if not hasattr(self, 'tp1_active') and close >= tp1:
                self.sell(size=int(round(self.position.size * 0.5)))
                print(f"🌙✨ PHASE 1 COMPLETE! Took