```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta
from pathlib import Path

# =====================
# 🌙 MOON DEV DATA PREP
# =====================
data_path = Path("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# 🌟 STRATEGY CLASS
# =====================
class DeltaBandBreakout(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_stddev = 2
    contraction_periods = 5
    contraction_threshold = 0.02  # 2%
    volume_surge_multiplier = 1.5  # 50% above average
    exit_bandwidth_ratio = 0.5
    exit_consecutive_bars = 3
    risk_per_trade = 0.02  # 2% of equity
    stop_loss_pct = 0.01  # 1%
    
    def init(self):
        # 🌙 Bollinger Bands Calculation
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_stddev,
            nbdevdn=self.bb_stddev,
            matype=0
        )
        self.I(lambda: self.bb_upper, name='BB_Upper')
        self.I(lambda: self.bb_middle, name='BB_Middle') 
        self.I(lambda: self.bb_lower, name='BB_Lower')

        # 🌊 Delta Volume (OBV)
        self.delta_volume = self.I(talib.OBV, self.data.Close, self.data.Volume, name='Delta_Volume')
        
        # 📈 Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')
        
        # Track entry bandwidth
        self.entry_bandwidth = None

    def next(self):
        # 🌙 Risk Management Check
        if len(self.trades) >= 3:
            last_three = [t for t in self.trades[-3:] if t.exit_price]
            if len(last_three) >=3 and all(t.pnl < 0 for t in last_three):
                print("🌙💔 MOON DEV ALERT: 3 consecutive losses - trading paused!")
                return

        if not self.position:
            # 🌟 Long Entry Conditions
            if self._check_contraction() and \
               self._negative_divergence() and \
               self.data.Close[-1] > self.bb_upper[-1] and \
               self.data.Volume[-1] > self.volume_ma[-1] * self.volume_surge_multiplier:
                
                # 🎯 Position Sizing
                entry_price = self.data.Close[-1]
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - stop_loss
                
                band_width = self.bb_upper[-1] - self.bb_lower[-1]
                position_size = (risk_amount / risk_per_share) * (1/band_width)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    self.entry_bandwidth = band_width
                    print(f"🌙🚀 MOON BLASTOFF! LONG {position_size} @ {entry_price} | SL: {stop_loss}")

            # 🌟 Short Entry Conditions (similar structure)
            
        else:
            # 🌙 Exit Conditions
            if self.position.is_long:
                current_bw = self.bb_upper[-1] - self.bb_lower[-1]
                
                # Volatility contraction exit
                if current_bw < self.entry_bandwidth * self.exit_bandwidth_ratio:
                    self.position.close()