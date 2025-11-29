Here's the complete fixed and debugged backtest code with Moon Dev optimizations:

```python
# 🌙 MOON DEV BACKTESTING SUITE - VOLATILITY SQUEEZE RSI STRATEGY 🚀

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
import pandas_ta as ta

# ========================
# DATA PREPROCESSING 🌍
# ========================
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and prepare moon data 🌕
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ========================
# STRATEGY IMPLEMENTATION 🚀
# ========================
class VolatilitySqueezeRSI(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    exit_periods = 480     # 5 days in 15m intervals (5*96=480) ⏳
    
    def init(self):
        # 🌗 CALCULATE CELESTIAL INDICATORS
        # 3-period RSI (convert days to 15m periods)
        self.rsi3 = self.I(talib.RSI, self.data.Close, timeperiod=3)
        
        # Price changes for volatility calculation
        price_changes = self.I(talib.MOM, self.data.Close, timeperiod=1)
        
        # 50-period volatility (standard deviation)
        self.volatility = self.I(talib.STDDEV, price_changes, timeperiod=50)
        
        # 10th percentile volatility threshold 🌌
        def rolling_pctl(arr, window, percentile):
            s = pd.Series(arr)
            return s.rolling(window).quantile(percentile/100)
        self.vol_pctl = self.I(rolling_pctl, self.volatility, 50, 10)
        
        self.entry_bar = 0  # Track entry bar for time exit

    def next(self):
        current_rsi = self.rsi3[-1]
        current_vol = self.volatility[-1]
        current_pctl = self.vol_pctl[-1]
        
        # 🌙 MOON DEV ENTRY CONDITIONS ✨
        if not self.position:
            if (current_rsi < 20) and (current_vol < current_pctl):
                entry_price = self.data.Close[-1]
                volatility_value = current_vol
                
                # Calculate cosmic risk levels 🌠
                sl_price = entry_price - 0.5 * volatility_value
                tp_price = entry_price + 1.0 * volatility_value
                
                # Calculate moon position size 🌝
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - sl_price
                
                if risk_per_share <= 0:
                    print("🌑 MOON DEV ERROR: Negative risk detected! Aborting launch 🚫")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, tp=tp_price)
                    self.entry_bar = len(self.data) - 1
                    print(f"🚀 MOON ENTRY 🌕 | Size: {position_size} | Price: {entry_price:.2f}")
                    print(f"   🌌 TP: {tp_price:.2f} | 🌑 SL: {sl_price:.2f}")

        # 🌗 EXIT CONDITIONS
        elif self.position:
            # Time-based exit
            if (len(self.data) - self.entry_bar) >= self.exit_periods:
                self.position.close()
                print("⏳ TIME EXIT: 5 days passed without reaching target 🌗")
            
            # RSI momentum exit
            if current_rsi > 50:
                self.position.close()
                print(f"📉 RSI EXIT: Current RSI {current_rsi:.1f} crossed 50 🌄")

# ========================
# LAUNCH BACKTEST 🚀🌙
# ========================
bt = Backtest(data, VolatilitySqueezeRSI, commission=.002, margin=1