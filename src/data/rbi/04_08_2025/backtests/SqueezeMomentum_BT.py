Here's the complete backtesting implementation for the SqueezeMomentum strategy:

```python
# 🌙 Moon Dev Squeeze Momentum Backtest 🌙
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy

# ======================
# DATA PREPARATION
# ======================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns 🌙
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

# ======================
# INDICATOR FUNCTIONS
# ======================
def bb_width(close, period=20, dev=2):
    upper, _, lower = talib.BBANDS(close, timeperiod=period, 
                                  nbdevup=dev, nbdevdn=dev)
    return upper - lower

# ======================
# STRATEGY IMPLEMENTATION
# ======================
class SqueezeMomentum(Strategy):
    def init(self):
        # 🌙 Core Indicators
        self.hma = self.I(pta.hma, self.data.Close, 20, name='HMA 20')
        self.bbw = self.I(bb_width, self.data.Close, 20, 2, name='BB Width')
        self.bbw_avg = self.I(talib.SMA, self.bbw, 50, name='BB Width Avg')
        self.vol_median = self.I(pta.median, self.data.Volume, 30, name='Vol Median')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, 14, name='ATR 14')
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready for launch! 🚀")

    def next(self):
        # Wait for enough data 🌙
        if len(self.data) < 100:
            return

        # ======================
        # ENTRY LOGIC
        # ======================
        if not self.position:
            # 🌙 Squeeze Conditions
            hma_up = self.hma[-1] > self.hma[-2]
            squeeze_on = self.bbw[-1] < 0.5 * self.bbw_avg[-1]
            vol_spike = self.data.Volume[-1] > 1.5 * self.vol_median[-1]

            if all([hma_up, squeeze_on, vol_spike]):
                # 🌙 Risk Management Calculation
                risk_percent = 0.01  # 1% equity risk
                risk_amount = self.equity * risk_percent
                atr_value = self.atr[-1] or 1  # Avoid division by zero
                
                position_size = risk_amount / (3 * atr_value)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.highest_high = self.data.High[-1]
                    print(f"🌙🚀 MOON BLASTOFF! Long {position_size} @ {self.data.Close[-1]:.2f}")
                    print(f"💎 HMA Rising | 📉 BB Squeeze | 📈 Volume Spike")

        # ======================
        # EXIT LOGIC
        # ======================
        if self.position:
            # Update trailing high 🌙
            self.highest_high = max(self.highest_high, self.data.High[-1])
            
            # Calculate dynamic stop 🌙
            stop_price = self.highest_high - 3 * self.atr[-1]
            
            if self.data.Low[-1] <= stop_price:
                self.position.close()
                print(f"🌙🛑 MOON LANDING! Exit @ {stop_price:.2f}")
                print(f"📉 Price: {self.data.Close[-1]:.2f} | 🔥 P/L: {self.position.pl:.2f}")

# ======================
# BACKTEST EXECUTION
# ======================
bt = Backtest(data, SqueezeMomentum, cash=1_000_000, exclusive_orders=True