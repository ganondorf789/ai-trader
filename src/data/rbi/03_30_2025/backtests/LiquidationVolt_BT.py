# 🌙 Moon Dev Backtest AI Implementation: LiquidationVolt Strategy ✨

# 1. Necessary Imports
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

# Data Handling with Moon Dev Standards 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationVolt(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌗 Core Indicators
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=2)
        
        # 🌊 Liquidation Indicators (Assumes 'long_liq'/'short_liq' columns exist)
        self.long_liq_avg = self.I(talib.SMA, self.data['long_liq'], 20)
        self.short_liq_avg = self.I(talib.SMA, self.data['short_liq'], 20)
        
        self.entry_bar = 0  # Track entry bar for time exit

    def next(self):
        price = self.data.Close[-1]
        upper_band = self.ema20[-1] + 2*self.atr[-1]
        lower_band = self.ema20[-1] - 2*self.atr[-1]

        # 🌑 Moon Entry Logic
        if not self.position:
            # Long Setup (Counter-trend in downtrend)
            if price < self.ema20[-1]:
                if (self.data['short_liq'][-1] > 2*self.short_liq_avg[-1] and
                    price > self.data.Close[-2] and
                    self.rsi[-1] > 30):
                    
                    sl = price - 1.5*self.atr[-1]
                    size = int(round((self.equity * self.risk_percent) / (price - sl))
                    self.buy(size=size, sl=sl)
                    print(f"🌙 MOON ALERT: LONG @ {price:.2f} | Size: {size} | SL: {sl:.2f} 🌠")

            # Short Setup (Counter-trend in uptrend)
            elif price > self.ema20[-1]:
                if (self.data['long_liq'][-1] > 2*self.long_liq_avg[-1] and
                    price < self.data.Close[-2] and
                    self.rsi[-1] < 70):
                    
                    sl = price + 1.5*self.atr[-1]
                    size = int(round((self.equity * self.risk_percent) / (sl - price)))
                    self.sell(size=size, sl=sl)
                    print(f"🌑 DARK SIDE: SHORT @ {price:.2f} | Size: {size} | SL: {sl:.2f} 🌌")

            self.entry_bar = len(self.data) if not self.position else self.entry_bar

        # 🌕 Exit Logic
        else:
            # Keltner Breakout Exit
            if price > upper_band or price < lower_band:
                self.position.close()
                print(f"🚀 STARSHIP EXIT @ {price:.2f} | Profit: {self.position.pl:.2f} 🌠")
            
            # Time-Based Exit (5 bars)
            elif len(self.data) - self.entry_bar >= 5:
                self.position.close()
                print(f"⏳ TIME WARP EXIT @ {price:.2f} | Bars Held: 5 🌈")

# 🌟 Run Moon Dev Backtest
bt = Backtest(data, LiquidationVolt, cash=1_000_000, commission=.002)
stats = bt.run()
print(stats)
print(stats._strategy)