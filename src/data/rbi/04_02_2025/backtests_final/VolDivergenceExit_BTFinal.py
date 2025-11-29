Here's the fully debugged and Moon Dev optimized backtest code with all technical issues fixed while preserving the original strategy logic:

```python
# 🌙 Moon Dev's VolDivergenceExit Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Clean and prepare the cosmic data 🚀
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

class VolDivergenceExit(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    obv_lookback = 5  # Divergence detection period 🌗
    
    def init(self):
        # 🌌 Cosmic Indicators
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        self.ema20 = self.I(talib.EMA, close, timeperiod=20, name='EMA20')
        self.atr20 = self.I(talib.ATR, high, low, close, timeperiod=20, name='ATR20')
        self.upper_kc = self.I(lambda e,a: e + 2.5*a, self.ema20, self.atr20, name='UpperKC')
        self.lower_kc = self.I(lambda e,a: e - 2.5*a, self.ema20, self.atr20, name='LowerKC')
        self.obv = self.I(talib.OBV, close, self.data.Volume, name='OBV')
        self.price_high = self.I(talib.MAX, high, timeperiod=5, name='PriceHigh5')
        self.obv_high = self.I(talib.MAX, self.obv, timeperiod=5, name='OBVHigh5')
        
        # 🌠 Trade tracking variables
        self.trailing_stop = None
        self.max_obv = None

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper_kc[-1]
        lower = self.lower_kc[-1]
        atr = self.atr20[-1]
        
        # 🌑 Long Entry Constellation
        if not self.position:
            # Check breakout and divergence
            if price > upper and len(self.price_high) >= 10:
                # Detect price/OBV divergence
                price_div = self.price_high[-1] > self.price_high[-6]
                obv_div = self.obv_high[-1] < self.obv_high[-6]
                
                if price_div and obv_div:
                    # Check volatility contraction
                    if (2.5*atr) < (0.5 * self.atr20[-20:].mean()):
                        print("🌌 Moon Alert: Volatility contraction - skipping trade!")
                        return
                    
                    # Calculate lunar position size 🌝
                    risk_amount = self.equity * self.risk_per_trade
                    risk_per_share = price - lower
                    if risk_per_share <= 0: 
                        print("🌑 Warning: Invalid risk per share calculation")
                        return
                        
                    size = int(round(risk_amount / risk_per_share))
                    
                    if size > 0:
                        self.buy(size=size, sl=lower)
                        self.trailing_stop = lower
                        self.max_obv = self.obv[-1]
                        print(f"🚀 LUNAR LIFTOFF! Entry: {price:.2f}, Size: {size}, Stop: {lower:.2f}")

        # 🌕 Managing open positions
        elif self.position.is_long:
            # Update trailing stop with cosmic winds 🌪️
            self.trailing_stop = max(self.trailing_stop, self.data.High[-1] - 2.5*atr)
            
            # Primary Exit: OBV confirmation 🌊
            if self.obv[-1] > self.max_obv:
                self.position.close()
                print(f"✨ STARGAZER EXIT! OBV Confirmation at {price:.2f}")
                return
                
            # Secondary Exit: Trailing stop