Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Ensure required columns with proper case
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolatilitySqueeze(Strategy):
    risk_pct = 2  # 2% risk per trade
    lookback_period = 252 * 2 * 96  # 2 years in 15m bars
    
    def init(self):
        # Check required columns exist
        if 'vix' not in self.data.df.columns or 'vvix' not in self.data.df.columns:
            raise ValueError("🌙 CRITICAL: Data must include 'vix' and 'vvix' columns")
        
        # VIX Bollinger Bands
        vix = self.data.df['vix']
        self.bb_upper = self.I(talib.BBANDS, vix, 20, 2, 2, name='BB_UPPER', which=0)
        self.vix_sma20 = self.I(talib.SMA, vix, timeperiod=20, name='VIX_SMA20')
        
        # VVIX 90th Percentile
        self.vvix_pct = self.I(self._rolling_percentile, self.data.df['vvix'], self.lookback_period, 90, name='VVIX_90th')
        
    def _rolling_percentile(self, series, window, q):
        return series.rolling(window).apply(lambda x: np.percentile(x, q))
    
    def next(self):
        # Moon Dev Debug Prints
        if len(self.data) % 1000 == 0:
            print(f"🌙 MOON DEV DEBUG: Bar {len(self.data)} | Close: {self.data.Close[-1]} | VIX: {self.data.df['vix'][-1]:.2f} | VVIX: {self.data.df['vvix'][-1]:.1f}")
        
        # Entry Logic
        if not self.position:
            vix_above_bb = self.data.df['vix'][-1] > self.bb_upper[-1]
            vvix_in_90th = self.data.df['vvix'][-1] >= self.vvix_pct[-1]
            
            if vix_above_bb and vvix_in_90th:
                # Risk Management Calculation
                equity = self.equity
                risk_amount = equity * self.risk_pct / 100
                entry_price = self.data.Close[-1]
                atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=14)[-1]
                
                # Use ATR-based stop (3x ATR)
                stop_loss = entry_price + 3*atr
                risk_per_unit = stop_loss - entry_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.sell(size=position_size)
                        print(f"🌙✨ MOON DEV ENTRY: Short {position_size} units at {entry_price:.2f} | Risk: {self.risk_pct}% | Equity: {equity:,.0f} 🚀")
        
        # Exit Logic
        else:
            # Exit on VIX cross below SMA20 (replaced crossover with direct comparison)
            if self.vix_sma20[-2] > self.data.df['vix'][-2] and self.vix_sma20[-1] < self.data.df['vix'][-1]:
                self.position.close()
                print(f"🚀🌙 MOON DEV EXIT: Profit {self.position.pl:.2f} | Price: {self.data.Close[-1]:.2f} ✨")

# Run Backtest
bt = Backtest(data, VolatilitySqueeze, cash=1_000_000, commission=.002