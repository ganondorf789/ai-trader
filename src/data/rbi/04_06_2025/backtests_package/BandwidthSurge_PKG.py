Here's the fixed code with all backtesting.lib references removed and replaced with Moon Dev approved alternatives:

```python
# 🌙 Moon Dev's BandwidthSurge Backtest Script ✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🚀 Data Preparation with Moon-Style Cleaning
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🌌 Cosmic Column Cleansing
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class BandwidthSurge(Strategy):
    def init(self):
        # 🌗 Lunar Indicators Configuration
        def bollinger_bandwidth(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
            
        self.bb_width = self.I(bollinger_bandwidth, self.data.Close)
        self.bb_low = self.I(talib.MIN, self.bb_width, timeperiod=20)
        self.volume_high = self.I(talib.MAX, self.data.Volume, timeperiod=50)
        
        def macd_hist(close):
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return hist
            
        self.macd_hist = self.I(macd_hist, self.data.Close)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌑🌒🌓🌔🌕🌖🌗🌘 Moon Dev Indicators Activated!")

    def next(self):
        # 🌠 Wait for cosmic alignment (sufficient data)
        if len(self.data) < 50:
            return

        # 🌪️ Bandwidth Expansion Exit Check
        if self.trades:
            current_bw = self.bb_width[-1]
            for trade in self.trades:
                if current_bw >= 2 * trade.tag['entry_bw']:
                    trade.close()
                    print(f"🌪️✨ Cosmic Expansion! Closing trade at {self.data.Close[-1]}")

        if not self.position:
            # 🌙 Long Entry Constellation
            bb_contraction = self.bb_width[-1] == self.bb_low[-1]
            vol_surge = self.data.Volume[-1] == self.volume_high[-1]
            macd_bull = (self.macd_hist[-2] < 0 and self.macd_hist[-1] > 0)  # Replaced crossover
            
            if bb_contraction and vol_surge and macd_bull:
                risk_amount = self.equity * 0.01
                atr_val = self.atr[-1]
                sl = self.data.Close[-1] - 1.5 * atr_val
                position_size = int(round(risk_amount / (self.data.Close[-1] - sl)))
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=sl,
                        tp=self.data.Close[-1] + 3*atr_val,
                        tag={'entry_bw': self.bb_width[-1]}
                    )
                    print(f"🚀🌕 BULLISH ASCENSION! Entry: {self.data.Close[-1]} | Size: {position_size}")

            # 🌑 Short Entry Eclipse
            macd_bear = (self.macd_hist[-2] > 0 and self.macd_hist[-1] < 0)  # Replaced crossunder
            if bb_contraction and vol_surge and macd_bear:
                risk_amount = self.equity * 0.01
                atr_val = self.atr[-1]
                sl = self.data.Close[-1] + 1.5 * atr_val
                position_size = int(round(risk_amount / (sl - self.data.Close[-1])))
                
                if position_size > 0:
                    self.sell(
                        size=position_size,
                        sl=sl,
                        tp=self.data.Close[-1]