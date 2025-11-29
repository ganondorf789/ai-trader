```python
# ZephyrMomentum Backtest Implementation 🌙🚀
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np

# Load and preprocess data 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

class ZephyrMomentum(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    stop_loss_pct = 0.005  # 0.5% stop loss
    max_positions = 3
    
    def init(self):
        # Core Indicators 🌙
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low,
                          close=self.data.Close,
                          volume=self.data.Volume,
                          length=20)
        
        # Z-Score Calculation ✨
        def calc_zscore(close, vwap):
            price_diff = close - vwap
            stddev = talib.STDDEV(price_diff, timeperiod=20)
            return np.divide(price_diff, stddev, where=stddev!=0)
        self.zscore = self.I(calc_zscore, self.data.Close, self.vwap)
        
        # MACD Histogram 🌙
        def calc_macd_hist(close):
            _, _, hist = talib.MACD(close, 
                                   fastperiod=12,
                                   slowperiod=26,
                                   signalperiod=9)
            return hist
        self.macd_hist = self.I(calc_macd_hist, self.data.Close)
        
        # TEMA Rocket Fuel 🚀
        self.tema = self.I(talib.TEMA, self.data.Close, timeperiod=8)
        
    def next(self):
        # Moon Dev Position Management ✨
        if len(self.positions) >= self.max_positions:
            return
            
        # Current Values 🌙
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        z = self.zscore[-1]
        macd = self.macd_hist[-1]
        tema = self.tema[-1]
        
        # Long Entry Constellation 🌌
        long_cond = all([
            crossover(self.data.Close, self.vwap),
            z > 1,
            macd > 0 and macd > self.macd_hist[-2],
            crossover(self.data.Close, self.tema)
        ])
        
        # Short Entry Comet Tail 🌠
        short_cond = all([
            crossover(self.vwap, self.data.Close),
            z < -1,
            macd < 0 and macd < self.macd_hist[-2],
            crossover(self.tema, self.data.Close)
        ])
        
        # Risk-Managed Execution 🌙
        if long_cond:
            sl = price * (1 - self.stop_loss_pct)
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / (price - sl)))
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=sl,
                        tp=price + 2*(price - sl))
                print(f"🌙✨ BLASTOFF LONG! Size: {position_size} | Entry: {price:.2f} 🚀")
                
        elif short_cond:
            sl = price * (1 + self.stop_loss_pct)
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / (sl - price)))
            if position_size > 0:
                self.sell(size=position_size,
                         sl=sl,
                         tp=price - 2*(sl - price))
                print(f"🌙✨ SHORT CIRCUIT! Size: {position_size} | Entry: {price:.2f} 🌌")
                
        # Galactic Exit Protocol 🌟
        for trade in self.positions:
            if trade.is_long and (self.zscore[-1] <= 0 or self.zscore[-