```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    
    # Clean column names 🧼
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

class VolterraDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    max_alloc = 0.05  # 5% max allocation 🚀
    
    def init(self):
        # Bollinger Bands (20,2) with TA-Lib 📈
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Band Width Calculations 🌗
        self.band_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.avg_band_width = self.I(talib.SMA, self.band_width, 20)
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        price = self.data.Close[-1]
        
        # VIX/VVIX Conditions (Assumes data contains these columns) 🌪️
        vix_inversion = (self.data['vix_front'][-1] > self.data['vix_second'][-1])
        vvix_pct = (self.data['vvix'][-1] - self.data['vvix'][-2])/self.data['vvix'][-2] if self.data['vvix'][-2] != 0 else 0
        vvix_spike = vvix_pct > 0.15
        
        # Bollinger Squeeze Condition 🌀
        squeeze = self.band_width[-1] < 0.5*self.avg_band_width[-1]
        
        # Moon Dev Entry Signals 🌗
        if not self.position:
            if all([vix_inversion, vvix_spike, squeeze]):
                if price > self.bb_middle[-1]:
                    self.enter_long()
                elif price < self.bb_middle[-1]:
                    self.enter_short()
        else:
            self.manage_exits()

    def enter_long(self):
        entry_price = self.data.Close[-1]
        stop_price = self.bb_lower[-1]
        risk_per_share = entry_price - stop_price
        
        if risk_per_share <= 0:
            print("🌙 Invalid Stop for Long! Skipping 🚫")
            return
        
        size = self.calc_position_size(risk_per_share, entry_price)
        if size > 0:
            self.buy(size=size, sl=stop_price, tag='LONG')
            print(f"🚀 LONG ENTRY: {size} units @ {entry_price} 🌕")

    def enter_short(self):
        entry_price = self.data.Close[-1]
        stop_price = self.bb_upper[-1]
        risk_per_share = stop_price - entry_price
        
        if risk_per_share <= 0:
            print("🌙 Invalid Stop for Short! Skipping 🚫")
            return
        
        size = self.calc_position_size(risk_per_share, entry_price)
        if size > 0:
            self.sell(size=size, sl=stop_price, tag='SHORT')
            print(f"🌑 SHORT ENTRY: {size} units @ {entry_price} 🌕")

    def calc_position_size(self, risk_per_share, entry_price):
        risk_amount = self.risk_pct * self.equity
        size = risk_amount / risk_per_share
        size = int(round(size))
        
        # Verify max allocation 🛑
        max_size = int((self.max_alloc * self.equity) // entry_price)
        return min(size, max_size)

    def manage_exits(self):
        position = self.position
        current_width = self.band_width[-1]
        
        # Take Profit Condition 💰
        if current_width >= self.avg_band_width[-1]:
            position.close()
            print(f"🎯 PROFIT