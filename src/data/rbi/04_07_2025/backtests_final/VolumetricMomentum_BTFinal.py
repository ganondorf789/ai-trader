Here's the fully debugged and optimized backtest code with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's VolumetricMomentum Backtest Script 🚀✨

import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data Preparation 🌍
try:
    data = pd.read_csv('BTC-USD-15m.csv')
    print("🌌 Cosmic data loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError("🚨 Lunar Alert: Data file not found. Please check the path.")

# Clean and format columns 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# DateTime processing 🌗
try:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime').sort_index()
    print("⏳ Time continuum stabilized!")
except KeyError:
    raise KeyError("🚨 Time warp detected: 'datetime' column missing")

class VolumetricMomentum(Strategy):
    def init(self):
        # 🌟 Indicator Calculation Zone
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX14')
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, 14, name='+DI')
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, 14, name='-DI')
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='SwingHigh')
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR14')
        self.upper_keltner = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr, name='UpperKeltner')
        self.lower_keltner = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr, name='LowerKeltner')
        print("✨ Celestial indicators initialized!")

    def next(self):
        # 🌙 Core Strategy Logic
        price = self.data.Close[-1]
        prev_swing_high = self.swing_high[-2] if len(self.swing_high) > 2 else 0
        
        # Entry Conditions Check ✅
        if not self.position:
            adx_valid = self.adx[-1] > 25
            trend_strength = self.plus_di[-1] > self.minus_di[-1]
            bullish_divergence = (self.data.Low[-1] < self.data.Low[-2]) and (self.obv[-1] > self.obv[-2])
            breakout_trigger = price > prev_swing_high
            
            if all([adx_valid, trend_strength, bullish_divergence, breakout_trigger]):
                # 🧮 Risk Management Calculation
                risk_amount = self.equity * 0.01  # 1% risk per trade
                entry_price = price
                stop_loss = max(self.lower_keltner[-1], 0.01)  # Ensure positive SL
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(
                            size=position_size,
                            sl=stop_loss,
                            tp=self.upper_keltner[-1]
                        )
                        print(f"🚀🌕 MOON LAUNCH! Long {position_size} units @ {entry_price:.2f}")
                        print(f"    🛡️ SL: {stop_loss:.2f} | 🎯 TP: {self.upper_keltner[-1]:.2f}")

        # Exit Conditions Check 🚪
        elif self.adx[-1] < 20: