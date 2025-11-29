```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

# Moon Dev Data Preparation Ritual 🌙✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse data columns with lunar magic 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeSpike(Strategy):
    risk_percent = 0.01  # 1% lunar dust risk per trade 💫
    
    def init(self):
        # 🌗 Bollinger Bandwidth Calculation
        self.sma20 = self.I(talib.SMA, self.data.Close, 20, name='SMA20')
        self.stddev = self.I(talib.STDDEV, self.data.Close, 20, 1, name='STDDEV20')
        self.bbw = self.I(lambda: (4 * self.stddev) / self.sma20, name='BBW')
        
        # 🌕 100-period BBW Median using Pandas TA
        self.bbw_median = self.I(lambda x: ta.median(x, length=100), 
                                self.bbw, name='BBW_MEDIAN100')
        
        # 📈 Volume Spike Detector
        self.volume_ma30 = self.I(talib.SMA, self.data.Volume, 30, name='VOL_MA30')
        
        # 🌌 Keltner Channel Configuration
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, 
                           self.data.Close, 14, name='ATR14')
        self.keltner_upper = self.I(lambda: self.ema20 + 2 * self.atr14,
                                   name='KELTNER_UPPER')
        
        # 🕒 Time Exit Tracking
        self.entry_bar = 0

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # 🌙 Entry Signal Configuration
            if len(self.bbw) < 100 or len(self.volume_ma30) < 30:
                return  # Wait for cosmic alignment 🌠
            
            bbw_condition = self.bbw[-1] < 0.5 * self.bbw_median[-1]
            vol_condition = self.data.Volume[-1] > 2 * self.volume_ma30[-1]
            
            if bbw_condition and vol_condition:
                # 🚀 Position Size Calculation
                risk_amount = self.equity * self.risk_percent
                atr = self.atr14[-1]
                stop_loss = price - 1.5 * atr
                risk_per_unit = 1.5 * atr
                
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    self.entry_bar = len(self.data) - 1
                    print(f"🌙✨ MOON DEV ENTRY ✨🌙 | Size: {position_size} | "
                          f"Price: {price:.2f} | SL: {stop_loss:.2f} | "
                          f"ATR: {atr:.2f}")
        else:
            # 🌕 Exit Conditions
            if self.position.is_long:
                # 🚀 Keltner Exit
                if price > self.keltner_upper[-1]:
                    self.position.close()
                    print(f"🚀🌕 MOON DEV KELTNER EXIT 🌕🚀 | "
                          f"Price: {price:.2f} | Upper Band: {self.keltner_upper[-1]:.2f}")
                
                # 🕒 Time-based Exit
                elif (len(self.data) - 1 - self.entry_bar) >= 3:
                    self.position.close()
                    print(f"⏳🌙 MOON DEV TIME EXIT 🌙⏳ | "
                          f