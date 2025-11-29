Here's the fixed backtest code with all technical issues resolved while preserving the original strategy logic:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT: VOLUMETRIC BREAKOUT STRATEGY �✨

# === REQUIRED IMPORTS ===
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# === DATA PREPARATION ===
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns according to Moon Dev specs 🌙
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

# === CORE STRATEGY CLASS ===
class VolumetricBreakout(Strategy):
    # Strategy parameters ✨
    ema_short = 50
    ema_long = 200
    adx_period = 14
    adx_threshold = 25
    vol_lookback = 20
    atr_period = 14
    risk_pct = 0.01
    atr_multiplier = 2
    
    def init(self):
        # === INDICATOR CALCULATIONS USING TALIB 🌙 ===
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema_short)
        self.ema200 = self.I(talib.EMA, self.data.Close, self.ema_long)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, self.vol_lookback)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Trade tracking variables 🛸
        self.trade_highest = None
        self.trail_stop = None

    def next(self):
        # === MOON DEV DEBUG PRINT 🌕 ===
        print(f"🌙 Processing {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f} | Equity: {self.equity:.2f}")
        
        # === ENTRY LOGIC ===
        if not self.position:
            # Golden Cross Check (2-bar confirmation)
            golden_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            
            # Momentum and Volume Check
            strong_trend = self.adx[-1] > self.adx_threshold
            volume_spike = self.data.Volume[-1] > self.vol_sma[-1]
            
            if golden_cross and strong_trend and volume_spike:
                # === RISK CALCULATION ===
                risk_amount = self.risk_pct * self.equity
                atr_value = self.atr[-1] or 0.0001  # Prevent zero division
                position_size = int(round(risk_amount / (self.atr_multiplier * atr_value)))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🚀🌕 MOON ENTRY! Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
                    # Initialize trail metrics
                    self.trade_highest = self.data.High[-1]
                    self.trail_stop = self.trade_highest - self.atr_multiplier * atr_value

        # === EXIT LOGIC ===
        elif self.position:
            # Update trailing stop 🌙
            self.trade_highest = max(self.trade_highest, self.data.High[-1])
            new_stop = self.trade_highest - (self.atr[-1] * self.atr_multiplier)
            self.trail_stop = max(self.trail_stop, new_stop)
            
            # Check stop breach
            if self.data.Low[-1] <= self.trail_stop:
                self.position.close()
                print(f"🌙✨ TRAILING STOP HIT! Exit: {self.data.Close[-1]:.2f}")