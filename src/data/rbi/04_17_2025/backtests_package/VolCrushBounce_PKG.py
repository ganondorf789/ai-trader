Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolCrushBounce(Strategy):
    def init(self):
        # 🌙 Cosmic Indicator Initialization
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevdn=2)[2], 
                             self.data.Close, name='BB_LOWER')
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20, name='SMA20')
        self.funding_percentile = self.I(ta.quantile, self.data.Funding_Rate, length=100, q=0.8, name='FUNDING_PCT')
        self.oi_peak = self.I(talib.MAX, self.data.Open_Interest, timeperiod=20, name='OI_PEAK')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        self.high_5 = self.I(talib.MAX, self.data.High, timeperiod=5, name='HIGH_5')
        print("🌙 MOON DEV INIT: Cosmic indicators activated! 🚀✨")

    def next(self):
        # 🌌 Check for existing positions
        if self.position:
            return

        # 📡 Check minimum data requirements
        if len(self.data) < 100:
            return

        # 🌪️ Calculate OI collapse
        current_oi = self.data.Open_Interest[-1]
        oi_drop = (self.oi_peak[-1] - current_oi)/self.oi_peak[-1] if self.oi_peak[-1] != 0 else 0

        # 🚀 Entry Conditions
        entry_signal = (
            self.data.Funding_Rate[-1] >= self.funding_percentile[-1],
            oi_drop >= 0.15,
            self.data.Close[-1] < self.bb_lower[-1]
        )

        if all(entry_signal):
            # 💰 Risk Management Calculations
            risk_pct = 0.01  # 1% equity risk
            atr_value = self.atr[-1]
            entry_price = self.data.Open[-1]
            stop_loss = entry_price + 1.5 * atr_value
            
            # 🧮 Position Sizing
            risk_amount = self.equity * risk_pct
            risk_per_unit = stop_loss - entry_price
            position_size = int(round(risk_amount / risk_per_unit))
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss)
                print(f"🌑🚀 MOON DEV SHORT ACTIVATED! 🌌\nEntry: {entry_price:.2f} | Size: {position_size} | Cosmic Stop: {stop_loss:.2f}")

        # 🌕 Exit Conditions
        for trade in self.trades:
            if trade.is_short:
                # Primary Exit
                if self.data.Close[-1] > self.sma20[-1]:
                    trade.close()
                    print(f"✨🌕 PRIMARY EXIT: Closed above SMA20 ({self.sma20[-1]:.2f})")
                
                # Emergency Exit
                if self.data.High[-1] > self.high_5[-2]:
                    trade.close()
                    print(f"🚨🌑 EMERGENCY EXIT: 5-period high breached ({self.high_5[-2]:.2f})")
                
                # Time Exit
                if len(self.data) - trade.entry_bar >= 96:
                    trade.close()
                    print(f"⏳🌓 TIME EXIT: 24-hour cosmic cycle completed")

# 🛰️ Data Preparation
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")

# 🧹 Data Cleaning
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'Funding_Rate',
    'open_interest': 'Open_Interest'
})
data.index = pd.to_datetime(data.datetime)

# 🌙 Backtest Execution
bt = Backtest(data, Vol