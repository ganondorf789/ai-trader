```python
# 🌙 MOON DEV BACKTESTING SCRIPT 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# ========================
# 🌌 DATA PREPROCESSING 🌌
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Cosmic Column Alignment 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ======================
# 🚀 GALACTIC STRATEGY 🚀
# ======================
class LiquidationBreakout(Strategy):
    ema_period = 20
    atr_period = 20
    atr_multiplier = 2.5
    risk_pct = 0.01
    max_holding_bars = 5
    funding_threshold = -0.001  # -0.1%
    
    def init(self):
        # 🌗 KELTNER CHANNEL CALCULATION 🌗
        def keltner_channels(high, low, close, ema_p, atr_p, mult):
            ema = talib.EMA(close, ema_p)
            atr = talib.ATR(high, low, close, atr_p)
            return ema + mult*atr, ema, ema - mult*atr
            
        self.keltner_upper, self.ema, self.keltner_lower = self.I(
            keltner_channels,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.ema_period,
            self.atr_period,
            self.atr_multiplier
        )
        
        # 📈 VOLUME CONFIRMATION 📈
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # 🌑 SWING LOW DETECTION 🌑
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("🌙 MOON DEV INDICATORS INITIALIZED 🌙")

    def next(self):
        # 🌠 MOON DEV DEBUG PRINT 🌠
        if len(self.data) % 100 == 0:
            print(f"🌕 PROCESSING BAR {len(self.data)} | Price: {self.data.Close[-1]:.2f} 🌕")

        # 🚀 LONG ENTRY CONDITION 🚀
        if not self.position:
            if (self.data.fundingrate[-1] < self.funding_threshold and
                self.data.Close[-1] > self.keltner_upper[-1] and
                self.data.Volume[-1] > self.volume_sma[-1]):
                
                # 🌙 RISK CALCULATION 🌙
                stop_price = min(self.swing_low[-1], self.keltner_lower[-1])
                risk_per_share = self.data.Close[-1] - stop_price
                
                if risk_per_share > 0:
                    position_size = (self.equity * self.risk_pct) / risk_per_share
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.entry_bar = len(self.data)
                        print(f"🚀🌙 LONG ENTRY 🚀🌙 | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {stop_price:.2f} | Funding: {self.data.fundingrate[-1]:.4f} 🌌")

        # ✨ EXIT CONDITIONS ✨
        else:
            current_bar = len(self.data)
            bars_held = current_bar - self.entry_bar
            
            # 🌓 MIDLINE EXIT (50%) 🌓
            if self.data.Close[-1] >= self.ema[-1]:
                if self.position.size > 0:
                    half_size = int(round(self.position.size * 0.5))
                    self.sell(size=half_size)
                    print(f"🌗🌙 MIDLINE EXIT 🌗🌙 | Sold {half_size} | Remaining: {self.position.size