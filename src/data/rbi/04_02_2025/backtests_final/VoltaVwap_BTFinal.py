Here's the fully debugged and Moon Dev themed backtest code with all technical issues fixed while preserving the original strategy logic:

```python
# 🌙 Moon Dev's VoltaVWAP Backtest Script ✨
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# 🚀 DATA PREPARATION 
print("🌕 Loading celestial market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print("🌗 Data purification complete! Ready for lunar analysis.")

class VoltaVWAPStrategy(Strategy):
    def init(self):
        # 🌗 INDICATOR SETUP
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        print("🌕 Moon Dev Indicators Activated: VWAP(20), ATR(14), RSI(14)")

    def next(self):
        # 🌑 CURRENT MARKET CONDITIONS
        price = self.data.Close[-1]
        vwap_val = self.vwap[-1]
        atr_val = self.atr[-1]
        rsi_val = self.rsi[-1]
        
        upper_band = vwap_val + 2 * atr_val
        lower_band = vwap_val - 2 * atr_val

        # 🌔 DEBUG DISPLAY
        print(f"\n🌙 Price: {price:.2f} | VWAP: {vwap_val:.2f} ± {2*atr_val:.2f}")
        print(f"✨ Upper: {upper_band:.2f}, Lower: {lower_band:.2f} | RSI: {rsi_val:.2f}")

        # 💰 RISK MANAGEMENT
        risk_pct = 0.01  # 1% risk per trade
        risk_amount = self.equity * risk_pct
        
        if not self.position:
            # 🚀 LONG ENTRY: Price > Upper Band & RSI <=70
            if price > upper_band and rsi_val <= 70:
                size = int(round(risk_amount / (1.5 * atr_val)))
                if size > 0:
                    sl = price - 1.5 * atr_val
                    print(f"\n🚀🌕 BULLISH BREAKOUT! Buying {size} units")
                    print(f"🔒 Stop Loss: {sl:.2f} ({1.5*atr_val:.2f} risk)")
                    self.buy(size=size, sl=sl)
            
            # 🚀 SHORT ENTRY: Price < Lower Band & RSI >=30
            elif price < lower_band and rsi_val >= 30:
                size = int(round(risk_amount / (1.5 * atr_val)))
                if size > 0:
                    sl = price + 1.5 * atr_val
                    print(f"\n🚀🌑 BEARISH BREAKOUT! Shorting {size} units")
                    print(f"🔒 Stop Loss: {sl:.2f} ({1.5*atr_val:.2f} risk)")
                    self.sell(size=size, sl=sl)
        else:
            # 🌓 EXIT CONDITIONS
            if self.position.is_long:
                # Manual RSI crossunder check
                if self.rsi[-2] > 70 and self.rsi[-1] <= 70:
                    print(f"\n🌗 LONG EXIT: RSI ({rsi_val:.2f}) crosses below 70")
                    self.position.close()
            
            elif self.position.is_short:
                # Manual RSI crossover check
                if self.rsi[-2] < 30 and self.rsi[-1] >= 30:
                    print(f"\n🌗 SHORT EXIT: RSI ({rsi_val:.2f})