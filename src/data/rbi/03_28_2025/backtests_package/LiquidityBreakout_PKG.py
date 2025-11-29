Here's the fixed code with all backtesting.lib imports removed and replaced with proper Moon Dev compliant implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDITYBREAKOUT STRATEGY 🚀

# REQUIRED IMPORTS
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# DATA PREPROCESSING
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and drop unnamed columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping ✨
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidityBreakout(Strategy):
    risk_percentage = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 SWING HIGH/LOW DETECTION (20-period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING LOW')
        
        # ✨ VOLUME DELTA SURGE INDICATOR (2x 10-period avg)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=10, name='VOL MA')
        
        # 🚀 BOLLINGER BANDS WITH WIDTH CALCULATION
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_width = self.I(lambda close: (self.bb_upper - self.bb_lower)/self.bb_upper, name='BB WIDTH')
        
        # 🌙 VOLATILITY EXPANSION FILTER
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! Ready for launch sequence...")

    def next(self):
        # Moon-themed debug prints 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} - Price: {self.data.Close[-1]:.2f}")
        
        if not self.position:
            # 🚀 LONG ENTRY CONDITIONS
            entry_conditions = [
                (self.data.High[-2] < self.swing_high[-2] and self.data.High[-1] > self.swing_high[-1]),  # Bullish breakout trigger
                self.data.Volume[-1] > 2 * self.vol_ma[-1],  # Volume surge
                self.bb_width[-1] > 0.05  # Simplified volatility filter
            ]
            
            if all(entry_conditions):
                # 🌙 RISK MANAGEMENT CALCULATIONS
                risk_amount = self._equity * self.risk_percentage
                sl_level = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl_level
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_level = self.data.Close[-1] + 3 * self.atr[-1]
                    
                    print(f"🚀🌙 MOON DEV BREAKOUT DETECTED! Buying {position_size} units")
                    print(f"   ENTRY: {self.data.Close[-1]:.2f} | SL: {sl_level:.2f} | TP: {tp_level:.2f}")
                    self.buy(size=position_size, sl=sl_level, tp=tp_level)

# RUN BACKTEST 🌙
bt = Backtest(data, LiquidityBreakout, cash=1_000_000, commission=.002)
stats = bt.run()

# PRINT FULL RESULTS ✨
print("\n🌙✨ MOON DEV FINAL