```python
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltrendFlip(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    volume_surge_multiplier = 2
    
    def init(self):
        # SuperTrend Indicator (10,3)
        supertrend = pta.supertrend(self.data.High, self.data.Low, self.data.Close, 10, 3)
        self.super_value = self.I(lambda: supertrend['SUPERT_10_3.0'], name='SuperTrend')
        self.super_dir = self.I(lambda: supertrend['SUPERTd_10_3.0'], name='SuperTrend_Dir')
        
        # Elder Force Index components
        self.force_index = self.I(talib.EMA, (self.data.Close.diff() * self.data.Volume).values, 13, name='EFI')
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 10, name='Volume_MA10')
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
    def next(self):
        if len(self.data) < 3:
            return
        
        current_dir = self.super_dir[-1]
        prev_dir = self.super_dir[-2]
        price = self.data.Close[-1]
        
        # Volume surge detection
        vol_surge = (self.data.Volume[-1] > self.volume_surge_multiplier * self.vol_ma[-1]) or \
                    (self.data.Volume[-2] > self.volume_surge_multiplier * self.vol_ma[-2])
        
        # Entry logic
        if not self.position:
            if current_dir == 1 and prev_dir == -1 and vol_surge:
                self.long_entry(price)
            elif current_dir == -1 and prev_dir == 1 and vol_surge:
                self.short_entry(price)
        else:
            self.check_exits(price)
            
    def long_entry(self, price):
        atr_val = self.atr[-1]
        stop_price = price - self.atr_multiplier * atr_val
        risk_amount = self.equity * self.risk_per_trade
        position_size = int(round(risk_amount / (price - stop_price)))
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_price)
            print(f"🌙✨🚀 BULLISH Moon Alignment! Long {position_size} @ {price:.2f} | SL: {stop_price:.2f}")
            
    def short_entry(self, price):
        atr_val = self.atr[-1]
        stop_price = price + self.atr_multiplier * atr_val
        risk_amount = self.equity * self.risk_per_trade
        position_size = int(round(risk_amount / (stop_price - price)))
        
        if position_size > 0:
            self.sell(size=position_size, sl=stop_price)
            print(f"🌙✨🚀 BEARISH Moon Flip! Short {position_size} @ {price:.2f} | SL: {stop_price:.2f}")
            
    def check_exits(self, price):
        current_dir = self.super_dir[-1]
        
        if self.position.is_long:
            # SuperTrend flip exit
            if current_dir == -1:
                self.position.close()
                print(f"🌙🌗 Moon Phase Shift! Closing long @ {price:.2f}")
            # Trailing stop check
            else:
                trail_stop = self.data.Close[-1] - self.atr_multiplier * self.atr[-1]
                if self.data.Low[-1] <= trail