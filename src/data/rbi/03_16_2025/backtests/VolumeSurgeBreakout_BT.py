```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumeSurgeBreakout(Strategy):
    ema_period = 50
    atr_period = 14
    swing_period = 20
    volume_window = 10
    risk_pct = 0.01
    
    def init(self):
        # Moon-powered indicators 🌗✨
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.resistance = self.I(talib.MAX, self.data.High, self.swing_period)
        self.support = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.volume_window)
        
        print("🌙✨ Moon Dev Indicators Activated! EMA50 | ATR14 | Swing Levels | Volume MA")

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Moon Dev status pulse 🌙💓
        if len(self.data) % 100 == 0:
            print(f"🌙 Pulse Check | Price: {price:.2f} | EMA50: {self.ema50[-1]:.2f} | ATR: {atr:.2f}")

        if not self.position:
            # Long entry constellation 🌌🚀
            if (price > self.resistance[-1] and
                self.data.Volume[-1] > 2*self.vol_ma[-1] and
                price > self.ema50[-1]):
                
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / atr)) if atr else 0
                if size:
                    sl = price - atr
                    tp = price + 1.5*atr
                    self.buy(size=size, sl=sl, tag='LONG')
                    print(f"🚀🌕 BLASTOFF LONG! Size: {size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

            # Short entry nebula 🌑📉    
            elif (price < self.support[-1] and
                  self.data.Volume[-1] > 2*self.vol_ma[-1] and
                  price < self.ema50[-1]):
                  
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / atr)) if atr else 0
                if size:
                    sl = price + atr
                    tp = price - 1.5*atr
                    self.sell(size=size, sl=sl, tag='SHORT')
                    print(f"🌑💫 BLACK HOLE SHORT! Size: {size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

        else:
            # Trailing stardust for longs ✨🚀
            if self.position.is_long:
                trail = self.data.High[-1] - 0.5*atr
                self.orders[0].sl = max(self.orders[0].sl, trail)
                
            # Trailing dark matter for shorts 🌑📉    
            elif self.position.is_short:
                trail = self.data.Low[-1] + 0.5*atr
                self.orders[0].sl = min(self.orders[0].sl, trail)

# Launch backtest sequence 🌙🚀
bt = Backtest(data, VolumeSurgeBreakout, cash=1_000_000, exclusive_orders=True)
stats = bt.run()
print("\n" + "="*50)