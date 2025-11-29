Here's the debugged and optimized version of your BandwidthSpike strategy with Moon Dev enhancements: 🌙✨

```python
# 🌙 Moon Dev Backtest Implementation for BandwidthSpike Strategy 🚀
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# =====================
# DATA PREPARATION 🌐
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and standardize columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# =====================
# STRATEGY CLASS 🚀🌙
# =====================
class BandwidthSpike(Strategy):
    risk_per_trade = 0.02  # 2% equity risk per trade 🌙
    
    def init(self):
        # 🌈 Indicator Calculation using TA-Lib
        def bbw_calculator(close):
            upper, middle, lower = talib.BBANDS(
                close, 
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            return (upper - lower) / middle
        
        # 🚀 Core Indicators
        self.bbw = self.I(bbw_calculator, self.data.Close)
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=10)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # 🌙 Trackers for position management
        self.highest_high = None

    def next(self):
        # 🌙 Moon Dev Debug Prints ✨
        if len(self.data) % 500 == 0:
            current_atr = self.atr[-1] if len(self.atr) > 0 else 0
            current_bbw = self.bbw[-1] if len(self.bbw) > 0 else 0
            print(f"🌙 Moon Pulse | Bar {len(self.data)} | BBW: {current_bbw:.4f} | ATR: {current_atr:.2f} | Equity: {self.equity:,.2f} ✨")
        
        # 🚀 Entry Logic
        if not self.position:
            if len(self.bbw) >= 10 and len(self.volume_ma) >= 20:  # Ensure enough data
                if (self.bbw[-1] == self.bbw_low[-1]) and (self.data.Volume[-1] > self.volume_ma[-1]):
                    # 🌈 Risk Management Calculation
                    risk_amount = self.equity * self.risk_per_trade
                    atr_value = self.atr[-1] if len(self.atr) > 0 else 0
                    
                    if atr_value > 0:
                        position_size = risk_amount / (2 * atr_value)
                        position_size = int(round(position_size))  # Ensure whole units
                        
                        if position_size > 0:
                            self.buy(size=position_size)
                            self.highest_high = self.data.High[-1]
                            print(f"🚀 Lunar Launch! Entry @ {self.data.Close[-1]:.2f} | Size: {position_size} ✨")
        
        # 🌙 Exit Logic
        elif self.position.is_long:
            # Update trailing stop
            self.highest_high = max(self.highest_high, self.data.High[-1])
            atr_value = self.atr[-1] if len(self.atr) > 0 else 0
            stop_price = self.highest_high - 2 * atr_value
            
            if self.data.Low[-1] < stop_price:
                self.sell(size=self.position.size)
                print(f"🌑 Moon Retreat! Exit @ {self.data.Close[-1]:.2f} | Trailing Stop: {stop_price:.2f} ✨")

# =