```python
# 🌙 MOON DEV BACKTESTING SCRIPT - MOMENTUMBREAKOUTEMA STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# 🌌 DATA PREPARATION - MOON-READY FORMAT ✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌠 PROPER COLUMN MAPPING - MOON STANDARDS 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data.sort_index(inplace=True)

class MomentumBreakoutEMA(Strategy):
    ema_short = 50
    ema_long = 200
    adx_period = 14
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade 🌕
    trail_pct = 0.05  # 5% trailing stop 🛑

    def init(self):
        # 🌟 INDICATORS USING TA-LIB THROUGH MOON WRAPPERS 🌙
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_short)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_long)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        
        print("🌙 MOON INDICATORS INITIALIZED: EMA50/200, ADX14, ATR14, 20-period Swing Highs ✨")

    def next(self):
        # 🌑 MOON CYCLE CHECK - ONLY TRADE WHEN NOT IN POSITION 🌓
        if not self.position:
            # 🌠 GOLDEN CROSS DETECTION SYSTEM 🚀
            ema_cross = crossover(self.ema50, self.ema200)[-1]
            
            # 💫 ADX STRENGTH FILTER ✨
            adx_strong = self.adx[-1] > 25 and self.adx[-1] > self.adx[-2]
            
            # 🌕 BREAKOUT CONFIRMATION SYSTEM 🌗
            price_breakout = self.data.Close[-1] > self.swing_high[-1]
            
            if ema_cross and adx_strong and price_breakout:
                # 🌙 RISK MANAGEMENT CALCULATIONS 💰
                entry_price = self.data.Close[-1]
                atr_value = self.atr[-1]
                stop_loss = entry_price - 2 * atr_value
                risk_amount = self.risk_pct * self.equity
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🚀 MOON ENTRY 🌕 | Price: {entry_price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | Equity: {self.equity:,.0f} ✨")
        else:
            # 🌗 TRAILING STOP SYSTEM - MOON PHASE LOCK 🔒
            current_high = self.data.High[-1]
            if current_high > self.position.entry_price:
                trail_price = current_high * (1 - self.trail_pct)
                if trail_price > self.position.sl:
                    self.position.sl = trail_price
                    print(f"🌓 TRAILING STOP ADJUSTED 🌗 | New SL: {trail_price:.2f} | Current High: {current_high:.2f}")

    def notify_trade_exit(self, exit_price: float, size: float, is_loss