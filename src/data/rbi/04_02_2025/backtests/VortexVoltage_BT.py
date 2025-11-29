```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexVoltage(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    adx_period = 14
    vi_period = 14
    swing_period = 20
    
    def init(self):
        # Vortex Indicator Calculation 🌪️
        def calculate_vi_plus():
            vi_plus, _ = ta.vortex(self.data.High, self.data.Low, self.data.Close, length=self.vi_period)
            return vi_plus.fillna(0)
        
        def calculate_vi_minus():
            _, vi_minus = ta.vortex(self.data.High, self.data.Low, self.data.Close, length=self.vi_period)
            return vi_minus.fillna(0)
            
        self.vi_plus = self.I(calculate_vi_plus, name='VI+')
        self.vi_minus = self.I(calculate_vi_minus, name='VI-')

        # Volume-Weighted ADX Calculation 📊
        def calculate_vwadx():
            adx = talib.ADX(self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
            return talib.SMA(adx * self.data.Volume, timeperiod=self.adx_period)
            
        self.vwadx = self.I(calculate_vwadx, name='VWADX')

        # Volume MA and Swing Levels 📈
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.swing_period, name='Vol MA')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing Low')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing High')

    def next(self):
        # Moon Dev Debug Prints 🌙
        if len(self.data) % 500 == 0:
            print(f"\n🌙 MOON DEV STATUS UPDATE @ {self.data.index[-1]}")
            print(f"   Balance: ${self.equity:,.2f} | Price: ${self.data.Close[-1]:.2f}")
            print(f"   VI+:{self.vi_plus[-1]:.2f} VI-:{self.vi_minus[-1]:.2f}")
            print(f"   VWADX:{self.vwadx[-1]:.2f} Vol:{self.data.Volume[-1]:.2f} MA:{self.vol_ma[-1]:.2f}")

        if not self.position:
            # Long Entry Logic 🚀
            if (crossover(self.vi_plus, self.vi_minus) and
                self.vwadx[-1] > 25 and
                self.data.Volume[-1] > self.vol_ma[-1]):
                
                sl = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl
                if risk_per_share > 0:
                    size = int(round((self.risk_pct * self.equity) / risk_per_share))
                    print(f"\n🌕 MOONSHOT LONG! Size: {size} | Entry: {self.data.Close[-1]:.2f}")
                    self.buy(size=size, sl=sl, tp=self.data.Close[-1] + 2*risk_per_share)

            # Short Entry Logic 🌑
            elif (crossover(self.vi_minus, self.vi_plus) and
                  self.vwadx[-1] > 25 and
                  self.data.Volume[-1] > self.vol_ma[-1]):
                  
                sl = self.swing_high[-1]
                risk_per_share = sl - self.data.Close[-1]
                if risk_per_share > 0:
                    size