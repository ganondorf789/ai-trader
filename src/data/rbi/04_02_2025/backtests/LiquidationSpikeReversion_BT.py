Here's the complete backtesting implementation for the LiquidationSpikeReversion strategy:

```python
# 🌙 MOON DEV BACKTESTING ENGINE v1.0 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ========== DATA PREPROCESSING ==========
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

# Verify strategy-specific columns exist 🚨
assert all(col in data.columns for col in ['funding_rate', 'liquidation_heatmap']), \
       "Missing required columns: funding_rate, liquidation_heatmap"

# ========== STRATEGY IMPLEMENTATION ==========
class LiquidationSpikeReversion(Strategy):
    def init(self):
        # 🌙 Funding Rate Indicators
        self.funding_ma = self.I(talib.SMA, self.data.funding_rate, 672)  # 7D SMA
        self.funding_std = self.I(talib.STDDEV, self.data.funding_rate, 672)
        
        # 💥 Liquidation Heatmap Analysis
        self.heatmap_30d_max = self.I(talib.MAX, self.data.liquidation_heatmap, 2880)  # 30D MAX
        
        # 📊 4H VWAP Calculation using TA-Lib
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        tpv = typical_price * self.data.Volume
        self.cumulative_tpv = self.I(talib.SUM, tpv, 16)  # 4H window
        self.cumulative_vol = self.I(talib.SUM, self.data.Volume, 16)
        self.vwap = self.I(lambda: self.cumulative_tpv / self.cumulative_vol, name='VWAP_4H')
        
        self.entry_bar = 0  # 🕒 Timeout counter

    def next(self):
        current_close = self.data.Close[-1]
        
        # 🌙 MOON DEV ENTRY LOGIC
        if not self.position:
            # Calculate current conditions
            funding_spike = (self.data.funding_rate[-1] < (self.funding_ma[-1] - 2*self.funding_std[-1]))
            heatmap_surge = (self.data.liquidation_heatmap[-1] >= 0.95*self.heatmap_30d_max[-1])
            below_vwap = current_close < self.vwap[-1]
            
            if funding_spike and heatmap_surge and below_vwap:
                # 🎯 Calculate risk parameters
                vwap_price = self.vwap[-1]
                distance = vwap_price - current_close
                
                if distance > 0:  # Valid trade setup
                    risk_percent = 0.01  # 1% risk
                    stop_loss = current_close - 1.5*distance
                    risk_per_share = current_close - stop_loss
                    position_size = int(round((self.equity * risk_percent) / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=stop_loss,
                                tp=vwap_price)
                        self.entry_bar = len(self.data)-1
                        print(f"🌙🚀 MOON DEV LONG ENTRY {current_close:.2f} | "
                              f"TP: {vwap_price:.2f} ({distance:.2f} points) | "
                              f"Size: {position_size} contracts")

        # 🛑 EMERGENCY EXIT CONDITIONS
        else:
            # Funding rate flipped positive
            if self.data.funding_rate[-1] > 0:
                self.position.close()
                print(f"🌙💸 FUNDING FLIP EXIT | Price: {current_close:.2f}")
            
            # Liquidation heatmap collapsed
            elif self.data.liquidation_heatmap[-1] < 0.5*self.heatmap_30d_max[-1]:
                self.position.close()
                print(f"🌙🔥