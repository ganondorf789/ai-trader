I'll fix the code by removing all backtesting.lib imports and replacing crossover/crossunder functions with manual checks. Here's the corrected version with Moon Dev's cosmic theme:

```python
# 🌙 Moon Dev's Liquidity Divergence Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preparation moon-style 🌕
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Convert index to datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# Load cosmic data 🌌
data = prepare_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

class LiquidityDivergence(Strategy):
    # Strategy parameters 🌗
    atr_period = 14
    swing_period = 20
    vol_ma_period = 20
    risk_pct = 0.01  # 1% risk per trade
    volume_surge_mult = 2
    volatility_threshold = 0.015  # 1.5% of price
    
    def init(self):
        # Cosmic indicators initialization 🌠
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='🌕 Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='🌑 Swing Low')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='🌀 ATR')
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period, name='📈 Vol MA')
        
        # Moon tracking variables 🌙
        self.entry_bar = None
        self.surge_volume = None

    def next(self):
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        
        # Cosmic position management 🌍
        if self.position:
            # Check volume exit condition 🌊
            if self.entry_bar and (len(self.data) - self.entry_bar) <= 3:
                if current_vol < self.surge_volume * 0.5:
                    self.position.close()
                    print(f"🌙✨ VOLUME COLLAPSE! Exiting at {current_close:.2f}")
                    self.entry_bar = None
                    return
            return

        # Calculate lunar conditions 🌘
        volatility_ok = current_atr < current_close * self.volatility_threshold
        volume_surge = current_vol > vol_ma * self.volume_surge_mult
        
        if volatility_ok and volume_surge:
            # Long entry constellation 🌟
            if current_close > self.swing_high[-1]:
                sl = current_close - 0.5 * current_atr
                tp = current_close + 1.5 * current_atr
                risk_amount = self.risk_pct * self.equity
                position_size = int(round(risk_amount / (current_close - sl)))
                
                self.buy(size=position_size, sl=sl, tp=tp)
                self.entry_bar = len(self.data)
                self.surge_volume = current_vol
                print(f"🌙🚀 BULLISH BREAKOUT! Size: {position_size} | Entry: {current_close:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

            # Short entry constellation 🌠
            elif current_close < self.swing_low[-1]:
                sl = current_close + 0.5 * current_atr
                tp = current_close - 1.5 * current_atr
                risk_amount = self.risk_pct * self.equity
                position_size = int(round(risk_amount / (sl - current_close)))
                
                self.s