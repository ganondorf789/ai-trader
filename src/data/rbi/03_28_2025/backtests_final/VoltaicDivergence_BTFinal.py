Here's the debugged backtest code with Moon Dev themed improvements 🌙✨:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Loading Sequence 🌙
print("🌙✨ Initializing Moon Dev Backtest Environment...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and standardize data columns
print("🌙🧹 Cleaning cosmic data dust...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Standardize column names for backtesting
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)

# Convert to datetime and set index
print("🌙⏳ Aligning temporal dimensions...")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VoltaicDivergence(Strategy):
    adx_period = 14
    keltner_ema = 20
    keltner_atr = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        print("🌙🔮 Initializing celestial indicators...")
        # Core indicators
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.ema = self.I(talib.EMA, self.data.Close, self.keltner_ema)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.keltner_atr)
        
    def next(self):
        # Moon Dev safety check 🌙
        if len(self.data) < 20:
            print("🌙⚠️ Not enough data yet - waiting for 20 periods")
            return
            
        i = len(self.data)-1
        current_close = self.data.Close[i]
        
        # Calculate dynamic Keltner bands
        upper_band = self.ema[i] + 2 * self.atr[i]
        lower_band = self.ema[i] - 2 * self.atr[i]
        
        # ======== ENTRY LOGIC ========
        if not self.position:
            # Moon Dev Trend Check 🌙
            if self.adx[i-1] > 30:
                # Bullish divergence detection
                if (self.data.Low[i-1] < self.data.Low[i-2] and 
                    self.obv[i-1] > self.obv[i-2]):
                    if current_close > self.data.High[i-1]:
                        risk_amount = self.equity * self.risk_pct
                        sl_price = self.data.Low[i-1]
                        risk_per_share = self.data.Close[i] - sl_price
                        if risk_per_share > 0:
                            size = int(round(risk_amount / risk_per_share))
                            print(f"🌙🚀 BULLISH DIVERGENCE DETECTED! Buying {size} units @ {current_close:.2f}")
                            self.buy(size=size, sl=sl_price)
                            
                # Bearish divergence detection        
                elif (self.data.High[i-1] > self.data.High[i-2] and 
                      self.obv[i-1] < self.obv[i-2]):
                    if current_close < self.data.Low[i-1]:
                        risk_amount = self.equity * self.risk_pct
                        sl_price = self.data.High[i-1]
                        risk_per_share = sl_price - self.data.Close[i]
                        if risk_per_share > 0:
                            size = int(round(risk_amount / risk_per_share))
                            print(f"🌙🚀 BEARISH DIVERGENCE DETECTED! Shorting {size} units @ {current_close:.2f}")
                            self.sell(size=size, sl=sl_price)
        
        # ======== EXIT LOGIC ========
        if self.position.is_long and current_close > upper_band:
            print(f"🌙💎 LONG EXIT SIGNAL @ {current_close:.2f} (Upper Band Touch)")
            self.position.close()
            
        elif self.position.is_short and current_close