```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOL SQUEEZE BREAKOUT STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import os

# =====================
# DATA PREPARATION 🌌
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY CLASS 🌗
# =====================
class VolSqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% of capital
    position_size = None
    extreme_high = None
    extreme_low = None

    def init(self):
        # 🌗 BOLLINGER BANDS CALCULATION
        def upper_bb(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        
        def lower_bb(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return lower

        self.upper_band = self.I(upper_bb, self.data.Close, name='UPPER BB')
        self.lower_band = self.I(lower_bb, self.data.Close, name='LOWER BB')
        
        # 📊 VOLATILITY MEASURES
        self.bb_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band, name='BB WIDTH')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=126, name='BB WIDTH MIN')
        
        # 🌊 VOLUME CONFIRMATION
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=30, name='VOL MA')
        
        # 🛑 RISK MANAGEMENT TOOLS
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=14, name='ATR(14)')

    def next(self):
        # Skip calculation until all indicators are ready ⏳
        if len(self.data) < 126:
            return

        # 🌙 MOON DEV DEBUG PRINTS ✨
        print(f"\n🌕 DATE: {self.data.index[-1]} | PRICE: {self.data.Close[-1]:.2f}")
        print(f"🔍 BB WIDTH: {self.bb_width[-1]:.4f} | 6M LOW: {self.bb_width_min[-1]:.4f}")
        print(f"📈 VOLUME: {self.data.Volume[-1]:.2f} vs MA: {self.volume_ma[-1]:.2f}")
        print(f"⚡ ATR(14): {self.atr[-1]:.2f}")

        # ENTRY LOGIC 🚪
        if not self.position:
            # Core entry conditions 💥
            bb_squeeze = self.bb_width[-1] <= self.bb_width_min[-1]
            volume_surge = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
            upper_break = self.data.Close[-1] > self.upper_band[-1]
            lower_break = self.data.Close[-1] < self.lower_band[-1]

            if bb_squeeze and volume_surge and (upper_break or lower_break):
                # Calculate position size 🧮
                risk_amount = self.risk_per_trade * self.equity
                atr_value = self.atr[-1]
                
                if upper_break:
                    direction = 'LONG'
                    entry_price = self.data.Close[-1]
                    stop_loss = entry_price - 2 * atr_value
                else:
                    direction = 'SHORT'
                    entry_price = self.data.Close