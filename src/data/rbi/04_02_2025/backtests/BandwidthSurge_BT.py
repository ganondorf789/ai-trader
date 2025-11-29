```python
# 🌙 Moon Dev's BandwidthSurge Backtest Implementation 🚀
import pandas as pd
from backtesting import Strategy, Backtest
import talib

# ========================
# DATA PREPARATION 🌍✨
# ========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})[['Open', 'High', 'Low', 'Close', 'Volume']]

# ========================
# STRATEGY CLASS 🌗🔭
# ========================
class BandwidthSurge(Strategy):
    def init(self):
        # 🌗 Bollinger Band Components
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close)
        
        # 🌌 Bandwidth Calculation
        self.bandwidth = self.I(lambda u, l: (u - l)/talib.SMA(self.data.Close, 20), 
                              self.upper_band, self.lower_band)
        
        # 🔭 Volatility and Volume Indicators
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, 20)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, 20)

    def next(self):
        # 🌙 Moon Dev Debug Prints
        print(f"\n🌕 New Candle: {self.data.index[-1]} | Equity: {self.equity:,.2f} 🌕")
        
        if not self.position:
            # ========================
            # ENTRY LOGIC 🚀🌙
            # ========================
            vol_spike = self.data.Volume[-1] > 1.5 * self.volume_ma[-2]
            bandwidth_contraction = self.bandwidth[-1] <= self.bandwidth_low[-1]
            
            if bandwidth_contraction and vol_spike:
                # 🌌 Long Entry Condition
                if self.data.Close[-1] > self.upper_band[-1]:
                    risk_amount = self.equity * 0.01
                    position_size = int(round(risk_amount / self.atr[-1]))
                    if position_size > 0:
                        self.buy(size=position_size, 
                                tag={'entry_price': self.data.Close[-1],
                                     'atr_entry': self.atr[-1]})
                        print(f"🚀🌙 LONG ENTRY! Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
                
                # 🌑 Short Entry Condition
                elif self.data.Close[-1] < self.lower_band[-1]:
                    risk_amount = self.equity * 0.01
                    position_size = int(round(risk_amount / self.atr[-1]))
                    if position_size > 0:
                        self.sell(size=position_size,
                                 tag={'entry_price': self.data.Close[-1],
                                      'atr_entry': self.atr[-1]})
                        print(f"🌑🚀 SHORT ENTRY! Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
        
        else:
            # ========================
            # EXIT LOGIC ✨🌓
            # ========================
            entry_price = self.position.tag['entry_price']
            atr_entry = self.position.tag['atr_entry']
            
            # 🌈 Long Position Management
            if self.position.is_long:
                tp = entry_price + 2 * atr_entry
                sl = entry_price - atr_entry
                
                if self.data.Close[-1] >= tp:
                    self.sell(size=self.position.size)
                    print(f"🌕✨ LONG TP HIT! +{2*atr_entry:.2f