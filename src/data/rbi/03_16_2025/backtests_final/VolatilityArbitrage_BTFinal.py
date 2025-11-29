```python
# 🌙 Moon Dev's Volatility Arbitrage Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS
# =====================
class VolatilityArbitrage(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_period = 14
    spread_window = 20
    threshold = 1.5  # Z-score threshold
    
    def init(self):
        # 🌙 Volatility Indicators
        self.hist_vol = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                              timeperiod=self.atr_period, name='HV_14')
        
        # ✨ Spread Analysis
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.spread_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.spread_window)
        self.spread = self.I(lambda: self.swing_high - self.swing_low, name='Vol_Spread')
        
        # 🌗 Z-score Calculation
        spread_mean = self.I(talib.SMA, self.spread, timeperiod=50)
        spread_std = self.I(talib.STDDEV, self.spread, timeperiod=50)
        self.z_score = self.I(lambda: (self.spread - spread_mean) / spread_std, name='Z-Score')

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity
        
        # 🌑 Ensure indicators are ready
        if len(self.z_score) < 3 or np.isnan(self.z_score[-1]):
            return

        # =====================
        # ENTRY LOGIC
        # =====================
        if not self.position:
            # Long entry on negative Z-score extreme
            if self.z_score[-1] < -self.threshold:
                risk_amount = equity * self.risk_pct
                sl = price * 0.98  # 2% stop loss
                position_size = int(round(risk_amount / (price - sl)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl)
                    print(f"🚀 MOON SHOT LONG: {price:.2f} | Size: {position_size} | Equity: {equity:,.2f}")
            
            # Short entry on positive Z-score extreme
            elif self.z_score[-1] > self.threshold:
                risk_amount = equity * self.risk_pct
                sl = price * 1.02  # 2% stop loss
                position_size = int(round(risk_amount / (sl - price)))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl)
                    print(f"🌒 DARK SIDE SHORT: {price:.2f} | Size: {position_size} | Equity: {equity:,.2f}")

        # =====================
        # EXIT LOGIC
        # =====================
        else:
            # Close position when Z-score normalizes
            if (self.position.is_long and self.z_score[-1] >= -0.5) or \
               (self.position.is_short and self.z_score[-1] <= 0.5):
               
                self.position.close()
                print(f"💰 STAR GAZER EXIT: {price:.2f} | P&L: {self.position.pl:+,.2f}")

# =====================
# BACKTEST EXECUTION
# =====================
bt = Backtest(data, VolatilityArbitrage, 
             cash=1_000_000,
             commission=.002,
             exclusive_orders=True)

stats = bt.run()
print("\n" + "🌙✨ Moon Dev's Volatility Arbitrage Back