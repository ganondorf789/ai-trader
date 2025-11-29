Here's the fixed code with all backtesting.lib imports removed and replaced with proper Moon Dev implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUICLUSTER BREAKOUT STRATEGY 🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquiClusterBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    bb_width_threshold = 0.5
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 BOLLINGER BAND WIDTH CALCULATION
        def bb_width_calc(close):
            upper, middle, lower = talib.BBANDS(close, 
                                              timeperiod=self.bb_period,
                                              nbdevup=self.bb_dev,
                                              nbdevdn=self.bb_dev)
            return (upper - lower) / middle
            
        self.bb_width = self.I(bb_width_calc, self.data.Close, name='🌙 BB_WIDTH')
        
        # 🚀 VOLUME CONFIRMATION INDICATOR
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_MA')
        
        print("✨ STRATEGY INITIALIZED WITH MOON POWER 🌙")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # 🌙 GET LIQUIDATION CLUSTER LEVELS FROM DATA
        cluster_high = self.data['cluster_high'][-1]
        cluster_low = self.data['cluster_low'][-1]
        
        print(f"🌙 CURRENT CLUSTERS | High: {cluster_high:.2f}, Low: {cluster_low:.2f} | BB Width: {self.bb_width[-1]:.4f}")

        if not self.position:
            # 🚀 ENTRY LOGIC
            if (self.bb_width[-1] < self.bb_width_threshold and
                current_close > cluster_high and
                current_volume > self.vol_ma[-1]):
                
                # 🌙 RISK MANAGEMENT CALCULATION
                sl_price = self.data.Low[-1]  # Breakout candle low
                risk_per_share = current_close - sl_price
                position_size = (self.equity * self.risk_percent) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🚀🌙 LONG ENTRY @ {current_close:.2f} | Size: {position_size} | SL: {sl_price:.2f} ✨")
                    
        else:
            # 🌙 EXIT LOGIC
            entry_price = self.position.entry_price
            entry_cluster = self.data['cluster_high'][self.position.entry_bar]
            
            # PRIMARY EXIT: VOLATILITY EXPANSION
            if self.bb_width[-1] > self.bb_width_threshold * 1.5:
                self.position.close()
                print(f"🎯🌙 VOLATILITY EXIT @ {current_close:.2f} | BB Width: {self.bb_width[-1]:.4f}")
                
            # EMERGENCY EXIT: RETURN TO CLUSTER ZONE
            elif current_close < entry_cluster:
                self.position.close()
                print(f"🚨🌙 EMERGENCY EXIT @ {current_close:.2f} | Cluster: {entry_cluster:.2f}")

# 🌙 DATA PREPARATION
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🚀 CLEAN AND FORMAT DATA
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 MAP REQUIRED COLUMNS
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_map, inplace=True)

# 🚀 VERIFY LIQUIDATION CLUSTER DATA
required_clusters = ['cluster_high', 'cluster_low']
if not all(col in data.columns for col in required_clusters):
    raise ValueError("❌ MISSING LIQUIDATION CLUSTER