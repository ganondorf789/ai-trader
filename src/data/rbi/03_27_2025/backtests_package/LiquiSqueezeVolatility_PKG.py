Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed enhancements:

```python
# 🌙 Moon Dev's LiquiSqueeze Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# 🚀 Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🧹 Cleanse and Align Data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquiSqueezeVolatility(Strategy):
    # 🌌 Strategy Parameters
    bb_period = 20
    bb_dev = 2
    rsi_period = 14
    swing_window = 20
    bbw_lookback = 50
    bbw_threshold = 0.2
    risk_pct = 0.01
    rr_ratio = 2
    
    def init(self):
        # 🎯 Initialize Indicators
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )
        
        # 🔍 Volatility Metrics
        self.bb_width = self.I(
            lambda: (self.bb_upper - self.bb_lower) / self.bb_middle,
            name='BB Width'
        )
        
        # 📈 Momentum Indicator
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
        # 🏔️ Liquidation Zones
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # 📊 Trade Tracking
        self.entry_bbw = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        # 🌙 Moon Dev Debug Prints
        if len(self.data) % 100 == 0:
            print(f"🌕 Lunar Cycle Check: Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | BBW: {self.bb_width[-1]:.4f}")
        
        if self.position:
            # 🎯 Manage Open Position
            current_low = self.data.Low[-1]
            current_high = self.data.High[-1]
            
            # 🚀 Take Profit Check
            if current_low <= self.take_profit:
                print(f"✨✨ PROFIT TAKEOFF! Closing at {self.data.Close[-1]:.2f}")
                self.position.close()
            
            # 🌧️ Stop Loss Check
            elif current_high >= self.stop_loss:
                print(f"🌧️🌧️ RAIN CHECK! Stopped at {self.data.Close[-1]:.2f}")
                self.position.close()
            
            # 🌀 Partial Close on Volatility Expansion
            else:
                current_bbw = self.bb_width[-1]
                if current_bbw >= self.entry_bbw * 1.5:
                    print(f"🌀 PARTIAL CLOSE! BBW Expanded 50%")
                    self.position.close(0.5)
                    self.entry_bbw = current_bbw  # Update trailing BBW
        else:
            # 🛑 Entry Conditions Check
            if len(self.data) < self.bbw_lookback:
                return
            
            # 📉 Volatility Contraction
            bbw_window = self.bb_width[-self.bbw_lookback:]
            bbw_percentile = np.percentile(bbw_window, self.bbw_threshold * 100)
            volatility_cond = self.bb_width[-1] < bbw_percentile
            
            # 🏔️ Liquidation Zone Test
            price_cond = self.data.High[-1] >= self.swing_high[-1] * 0.995
            
            #