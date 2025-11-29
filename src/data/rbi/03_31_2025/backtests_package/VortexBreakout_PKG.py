Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev Backtest AI Implementation for VortexBreakout Strategy
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VortexBreakout(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade
    
    def init(self):
        # 🌙 Vortex Indicator Calculation
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        # Calculate Vortex using pandas_ta
        vortex = ta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda: vortex['VORTIC_14_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTICm_14_14'], name='VI-')
        
        # 🌀 Keltner Channels Calculation
        ema20 = talib.EMA(close, timeperiod=20)
        atr10 = talib.ATR(high, low, close, timeperiod=10)
        self.kc_upper = self.I(lambda: ema20 + 2*atr10, name='KC Upper')
        self.kc_lower = self.I(lambda: ema20 - 2*atr10, name='KC Lower')
        
        # 📈 ADX for trend filtering
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=14, name='ADX')
        
        print("🌙✨ VortexBreakout Strategy Initialized! Ready for cosmic profits! 🚀")

    def next(self):
        # 🌑 Current price reference
        price = self.data.Close[-1]
        
        # 🛑 Exit conditions for existing positions
        if self.position.is_long and price < self.kc_upper[-1]:
            self.position.close()
            print(f"🌙💫 Closing LONG at {price:.2f} (Returned inside KC Upper)")
            
        if self.position.is_short and price > self.kc_lower[-1]:
            self.position.close()
            print(f"🌙🌑 Closing SHORT at {price:.2f} (Returned inside KC Lower)")
        
        # 🚫 Skip entries if ADX <25 (weak trend)
        if self.adx[-1] < 25:
            return
        
        # 💰 Risk calculation
        equity = self.portfolio.equity
        risk_amount = equity * self.risk_per_trade
        
        # 🚀 Long Entry: VI+ crosses VI- & price > KC Upper
        if (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and price > self.kc_upper[-1]:
            sl_price = self.kc_lower[-1]
            risk_per_share = price - sl_price
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price)
                print(f"🌙🚀 LONG Signal! {position_size} units @ {price:.2f} | SL: {sl_price:.2f}")
        
        # 🌑 Short Entry: VI- crosses VI+ & price < KC Lower
        elif (self.vi_minus[-2] > self.vi_plus[-2] and self.vi_minus[-1] < self.vi_plus[-1]) and price < self.kc_lower[-1]:
            sl_price = self.kc_upper[-1]
            risk_per_share = sl_price - price
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.sell(size=position_size, sl=sl_price)
                print(f"🌙🌒 SHORT Signal! {position_size} units @ {price:.2f} | SL: {sl_price:.2f}")

# 🌙 Data Preparation
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# 🧹 Data Cleaning
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low':