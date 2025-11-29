Here's the fixed code with all Moon Dev package requirements implemented and debug prints added:

```python
# 🌙 Moon Dev's Volatility Pulse Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np  # Added for array operations

# 🚀 DATA PREPARATION 
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityPulse(Strategy):
    atr_period = 14
    risk_per_trade = 0.01  # 1% risk
    
    def init(self):
        # 🌗 INDICATOR CALCULATION (Pure TA-Lib implementation)
        print("🌙 Initializing Moon Dev's Volatility Pulse Strategy...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, self.atr_period, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, 20, name='ATR_MA')
        
        # 🌓 VIX Low Calculation (Pure TA-Lib implementation)
        if 'vix' in self.data.df.columns:
            self.vix_low = self.I(talib.MIN, self.data.df['vix'], 1920, name='VIX_LOW')  # 20 days in 15m
        else:
            print("⚠️ Warning: VIX data not found - using dummy values")
            self.vix_low = self.I(np.zeros_like, self.data.Close, name='VIX_LOW')
        
        self.trade_stops = {}  # 🌊 Trailing stop tracker
        print("🌙 Indicators initialized successfully!")

    def next(self):
        price = self.data.Close[-1]
        
        # 🌈 ENTRY LOGIC (No backtesting.lib used)
        if not self.position and len(self.atr) > 20:
            # Bullish ATR crossover detection (manual implementation)
            atr_cross = (self.atr[-2] <= self.atr_ma[-2]) and (self.atr[-1] > self.atr_ma[-1])
            
            # VIX condition check
            vix_ok = False
            if 'vix' in self.data.df.columns:
                vix_ok = self.data.df['vix'][-1] < self.vix_low[-1]
            
            if atr_cross and vix_ok:
                risk_amount = self.equity * self.risk_per_trade
                stop_dist = 2 * self.atr[-1]
                
                if stop_dist > 0:
                    size = int(round(risk_amount / stop_dist))
                    entry_price = self.data.Open[-1]
                    max_size = int(self.cash // entry_price)
                    size = min(size, max_size)
                    
                    if size > 0:
                        self.buy(size=size, tag='🌙 VOLATILITY BREAKOUT')
                        print(f"🚀 ENTRY | Size: {size} BTC | Entry: {entry_price:.2f} | Stop: {entry_price - stop_dist:.2f}")
                        self.trade_stops[self.position.id] = entry_price - stop_dist

        # 🌧️ EXIT LOGIC (Pure array-based implementation)
        for trade in self.trades:
            if trade.is_long:
                trail_stop = self.data.High[-1] - 2 * self.atr[-1]
                current_stop = self.trade_stops.get(trade.id, trail_stop)
                self.trade_stops[trade.id] = max(trail_stop, current_stop)
                
                if self.data.Low[-1] < self.trade_stops[trade.id]:
                    self.position.close()
                    print(f"🌧️ EXIT | Price: {price:.2f} | Stop: {self.trade_stops[trade.id]:.2f} | Profit: {trade.pl_pct:.2%}")
                    del self.trade_stops[trade.id]

# 🚀 BACKTEST EXECUTION
print("\n🌙 Starting Moon Dev's Volatility Pulse Backtest...