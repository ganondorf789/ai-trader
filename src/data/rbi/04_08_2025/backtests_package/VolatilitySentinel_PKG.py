Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
# 🌙 Moon Dev's VolatilitySentinel Backtest Implementation 🚀✨

from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data Preparation 🌍
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolatilitySentinel(Strategy):
    risk_pct = 0.02  # 🌑 2% risk per trade
    ema_period = 3
    median_period = 21
    vol_lookback = 20  # For realized volatility
    
    def init(self):
        # Indicator Calculation 🌗
        put_call = self.data.df['put_call_ratio']
        
        # Core Indicators
        self.ema_putcall = self.I(talib.EMA, put_call, timeperiod=self.ema_period, name='EMA3_PCR')
        self.median_putcall = self.I(lambda s: s.rolling(self.median_period).median(), 
                                   put_call, name='MED21_PCR')
        self.vix_spread = self.I(lambda: self.data.df['vix_near'] - self.data.df['vix_next'],
                                name='VIX_SPREAD')
        self.realized_vol = self.I(talib.STDDEV, self.data.Close, timeperiod=self.vol_lookback, 
                                 name='REALIZED_VOL')
        
        print("🌙✨ Volatility Sentinel Activated! Initial Indicators:")
        print(f"EMA({self.ema_period}) PCR | MED({self.median_period}) PCR | VIX Spread | {self.vol_lookback}-Period Vol")
        print("🌌 All indicators powered by Moon Dev's celestial calculations!")

    def next(self):
        if self.position:
            return  # 🛑 Existing position management

        # Entry Conditions Check 🔍 (using manual crossover detection)
        ema_cross = (self.ema_putcall[-2] < self.median_putcall[-2] and 
                     self.ema_putcall[-1] > self.median_putcall[-1])
        vix_confirmation = self.vix_spread[-1] > 0  # Backwardation
        
        if ema_cross and vix_confirmation:
            # Risk Management Calculation 🔒
            equity = self.equity
            risk_amount = equity * self.risk_pct
            entry_price = self.data.Close[-1]
            atr = self.realized_vol[-1]
            
            # Dynamic Stop Calculation 📏
            stop_price = entry_price + 2*atr
            risk_per_unit = stop_price - entry_price
            
            if risk_per_unit <= 0:
                print("⚠️🌘 Invalid Risk/Unit - Trade Aborted (Moon Dev Protection)")
                return
                
            position_size = int(round(risk_amount / risk_per_unit))
            
            if position_size <= 0:
                print("⚠️🌒 Zero Size Calculated - Trade Skipped (Moon Phase Safety)")
                return
                
            # Execute Short Entry 🩳
            print(f"🚀🌑 MOON DEV SHORT SIGNAL ACTIVATED!")
            print(f"🌕 Entry: {entry_price:.2f} | 🌗 Size: {position_size} | 🌓 Vol Stop: {stop_price:.2f}")
            print("🌙✨ Executing celestial trade alignment...")
            self.sell(size=position_size, sl=stop_price, tag='VolatilityStop')

# Backtest Execution 🔁
print("\n🌑🌒🌓🌔🌕 MOON DEV BACKTEST INITIATION 🌕🌔🌓🌒🌑")
bt = Backtest(data, VolatilitySentinel, 
             cash=1_000_000, commission=.002, 
             exclusive_orders=True)

stats = bt.run()
print("\n🌕🌕🌕 FINAL BACKTEST RESULTS 🌕🌕🌕")
print("✨ Celestial Performance Metrics ✨")
print(stats)
print(stats._strategy)
print("\n🌙