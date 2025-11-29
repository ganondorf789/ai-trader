Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
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

class KeltnerWilliamsBreakout(Strategy):
    def init(self):
        # Williams %R (3-day period)
        self.willr = self.I(talib.WILLR, self.data.High, self.data.Low, self.data.Close, timeperiod=288)
        
        # Keltner Channel components
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_keltner = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr20)
        
        # Bollinger Bands
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, multiple=True)
        
        # Volatility filter
        self.bb_width = self.I(lambda u, l: u-l, self.bb_upper, self.bb_lower)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=20)
        
        self.highest_high = 0

    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Moon Dev Status ✨ | Price: {price:.2f} | %R: {self.willr[-1]:.2f} | KC Upper: {self.upper_keltner[-1]:.2f}")
        
        if not self.position:
            # Entry conditions (replaced crossover with manual check)
            willr_cross = (self.willr[-2] < -80 and self.willr[-1] > -80)  # Bullish crossover of -80
            price_break = price > self.upper_keltner[-1]
            volatility_ok = self.bb_width[-1] <= 3*self.bb_width_avg[-1]
            
            if willr_cross and price_break and volatility_ok:
                # Risk calculations
                risk_amount = 0.02 * self.equity
                risk_per_share = price * 0.05  # 5% stop
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.highest_high = price
                    print(f"\n🚀 MOON DEV LAUNCH 🚀"
                          f"\nEntry Price: {price:.2f}"
                          f"\nSize: {position_size} BTC"
                          f"\nEquity: {self.equity:.2f} 🌕")
        
        else:
            # Update trailing stop
            self.highest_high = max(self.highest_high, self.data.High[-1])
            trailing_stop = self.highest_high * 0.95
            
            # Exit conditions
            if self.data.Low[-1] <= trailing_stop:
                self.position.close()
                print(f"\n🛑 MOON DEV LANDING 🛑"
                      f"\nExit Price: {price:.2f}"
                      f"\nProfit: {self.position.pl_usd:.2f}"
                      f"\nEquity: {self.equity:.2f} 🌙")
            
            elif self.data.High[-1] >= self.bb_upper[-1]:
                self.position.close()
                print(f"\n🎯 MOON DEV TARGET HIT 🎯"
                      f"\nExit Price: {price:.2f}"
                      f"\nProfit: {self.position.pl_usd:.2f