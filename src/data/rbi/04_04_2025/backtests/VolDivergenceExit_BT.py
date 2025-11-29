```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolDivergenceExit(Strategy):
    def init(self):
        # Add indicators using TA-Lib with self.I()
        self.I(talib.STDDEV, self.data.Close, timeperiod=10, name='HV_10')
        self.I(talib.SMA, self.data.HV_10, timeperiod=30, name='HV_30_Avg')
        
        # Calculate PVT using pandas_ta
        pvt = ta.pvt(close=self.data.Close, volume=self.data.Volume)
        self.I(lambda: pvt, name='PVT')
        
        # Swing high/low indicators
        self.I(talib.MAX, self.data.High, timeperiod=20, name='Price_Swing_High')
        self.I(talib.MAX, self.data.PVT, timeperiod=20, name='PVT_Swing_High')
        self.I(talib.MIN, self.data.Low, timeperiod=20, name='Price_Swing_Low')
        
        # 50-day SMA and ATR
        self.I(talib.SMA, self.data.Close, timeperiod=50, name='SMA_50')
        self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')

        self.entry_bar = 0  # Track entry bar for time exit

    def next(self):
        if not self.position:
            # Entry conditions check
            hv_cross = (self.data.HV_10[-1] < self.data.HV_30_Avg[-1] and 
                        self.data.HV_10[-2] >= self.data.HV_30_Avg[-2])
            
            price_divergence = (self.data.Price_Swing_High[-1] > self.data.Price_Swing_High[-2] and 
                               self.data.PVT_Swing_High[-1] < self.data.PVT_Swing_High[-2])
            
            confirmation = self.data.Close[-1] < self.data.Price_Swing_Low[-1]
            
            if hv_cross and price_divergence and confirmation:
                # Risk management calculations
                risk_pct = 0.01
                atr = self.data.ATR[-1]
                stop_loss = self.data.Price_Swing_High[-1] + 2*atr
                risk_amount = self.equity * risk_pct
                risk_per_share = stop_loss - self.data.Close[-1]
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.sell(size=position_size, sl=stop_loss, tag='🌑 Short Entry')
                    self.entry_bar = len(self.data) - 1
                    print(f"🌑 Moon Dev Signal! SHORT {position_size} @ {self.data.Close[-1]} 🚀 SL: {stop_loss:.2f}")
        else:
            # Exit conditions check
            if self.data.Low[-1] <= self.data.SMA_50[-1]:
                self.buy(size=self.position.size, tag='🌙 SMA Exit')
                print(f"🌙 SMA Touch Exit! Closed @ {self.data.Close[-1]} ✨")
            
            elif (self.data.HV_10[-1] > self.data.HV_30_Avg[-1] and 
                  self.data.HV_10[-2] <= self.data.HV_30_Avg[-2]):
                self.buy(size=self.position.size, tag='🚨 Vol Exit')
                print(f"🚨 Volatility Spike! Closed @ {self.data.Close[-1]} 🌪️")
            
            elif (len(self.data) - self.entry_bar) >= 5*96:  # 5 days in 15m bars
                self.buy(size=self.position.size, tag='⏰ Time Exit')
                print(f"⏰ 5-Day Timeout! Closed @ {self.data.Close[-1]} 🕒")

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close