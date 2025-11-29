Here's the complete backtesting implementation for the VolSqueezeBreakout strategy:

```python
# 🌙 Moon Dev's Volatility Squeeze Breakout Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import TimeFrame

class VolSqueezeBreakout(Strategy):
    donchian_period = 20
    vol_percentile_window = 30
    volume_median_window = 50
    
    def init(self):
        # Calculate Donchian Channel
        self.donchian_high = self.I(talib.MAX, self.data.High, self.donchian_period, name='Donchian High')
        self.donchian_low = self.I(talib.MIN, self.data.Low, self.donchian_period, name='Donchian Low')
        
        # Calculate Channel Width
        self.channel_width = self.I(lambda high, low: high - low, 
                                  self.donchian_high, self.donchian_low, 
                                  name='Channel Width')
        
        # Volatility Percentiles
        self.vol_p30 = self.I(lambda x: x.rolling(self.vol_percentile_window).quantile(0.3),
                            self.channel_width, name='Volatility 30th %ile')
        self.vol_p70 = self.I(lambda x: x.rolling(self.vol_percentile_window).quantile(0.7),
                            self.channel_width, name='Volatility 70th %ile')
        
        # Volume Median
        self.volume_median = self.I(lambda x: x.rolling(self.volume_median_window).median(),
                                  self.data.Volume, name='Volume Median')
        
        print("🌙✨ Moon Dev Strategy Initialized! Ready for Launch! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Get indicator values
        d_high = self.donchian_high[-1]
        d_low = self.donchian_low[-1]
        vol_p30 = self.vol_p30[-1]
        vol_p70 = self.vol_p70[-1]
        vol_med = self.volume_median[-1]
        c_width = self.channel_width[-1]

        # 🌙 Moon Dev Position Sizing Calculator
        def calculate_size(entry_price, stop_price):
            risk_percent = 0.01  # 1% risk per trade
            risk_amount = self.equity * risk_percent
            risk_per_share = abs(entry_price - stop_price)
            if risk_per_share == 0:
                return 0
            return int(round(risk_amount / risk_per_share))

        # 🛑 Exit Conditions
        if self.position.is_long:
            if current_close < d_high or c_width > vol_p70:
                self.position.close()
                print(f"🌙💎 Closing LONG at {current_close}: {'Reversal' if current_close < d_high else 'Vol Spike'}")

        elif self.position.is_short:
            if current_close > d_low or c_width > vol_p70:
                self.position.close()
                print(f"🌙💎 Closing SHORT at {current_close}: {'Reversal' if current_close > d_low else 'Vol Spike'}")

        # 🚀 Entry Conditions
        if not self.position:
            # Long Entry
            if (current_close > d_high and 
                c_width < vol_p30 and 
                current_volume > vol_med):
                
                size = calculate_size(current_close, d_low)
                if size > 0:
                    self.buy(size=size, sl=d_low)
                    print(f"🌙🚀 LONG ENTRY! Size: {size} @ {current_close}")
            
            # Short Entry
            elif (current_close < d_low and 
                  c_width < vol_p30 and 
                  current_volume > vol_med):
                
                size = calculate_size(current_close, d_high)
                if size > 0:
                    self.sell(size=size, sl=d_high)
                    print(f"🌙🚀 SHORT ENTRY! Size: {size} @ {current_close}")

# 🌙 Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])