```python
# 🌙✨ Moon Dev's VortexVolumeExpansion Backtest Script 🚀

# Required imports
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexVolumeExpansion(Strategy):
    def init(self):
        # 🌙 Step 1: Calculate Indicators using self.I()
        # Vortex Indicator (VI+ and VI-)
        self.vi_plus = self.I(self._calculate_vi_plus, name='VI+')
        self.vi_minus = self.I(self._calculate_vi_minus, name='VI-')
        
        # Volume 20-period MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume MA')
        
        # ATR for volatility (14-period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # ATR MA for exit condition (10-period)
        self.atr_ma = self.I(talib.SMA, self.atr, 10, name='ATR MA')
        
        print("🌙✨ Strategy Initialized with Vortex, Volume MA, and ATR Indicators!")

    def _calculate_vi_plus(self):
        # Calculate VI+ using pandas_ta
        vi_df = ta.vortex(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close, 
            length=14
        )
        return vi_df['VORTICSm_14']

    def _calculate_vi_minus(self):
        # Calculate VI- using pandas_ta
        vi_df = ta.vortex(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close, 
            length=14
        )
        return vi_df['VORTICSr_14']

    def next(self):
        # 🌙✨ Core Logic Executes Here
        if len(self.data) < 20:  # Warmup period
            return

        # Current values for debugging
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Entry Conditions
        vi_cross = crossover(self.vi_plus, self.vi_minus)
        volume_ok = current_volume > self.volume_ma[-1]
        atr_expanding = self.atr[-1] > self.atr[-2]

        # Moon Dev Entry Signal 🌙🚀
        if not self.position and vi_cross and volume_ok and atr_expanding:
            # Risk Management
            risk_percent = 0.01  # 1% risk per trade
            entry_price = current_close
            stop_loss = entry_price * 0.99  # 1% fixed SL
            
            # Position sizing calculation
            risk_amount = self.equity * risk_percent
            risk_per_share = entry_price - stop_loss
            position_size = risk_amount / risk_per_share
            position_size = int(round(position_size))  # Critical rounding
            
            if position_size > 0:
                self.buy(
                    size=position_size, 
                    sl=stop_loss,
                    tag="🌙✨ VortexVolumeExpansion Long"
                )
                print(f"🌙✨🚀 MOON DEV ENTRY: Long {position_size} units at {entry_price:.2f} | SL {stop_loss:.2f}")

        # Exit Conditions
        if self.position:
            # Primary Exit: VI- crosses above VI+
            exit_signal = crossover(self.vi_minus, self.vi_plus)
            # Secondary Exit: ATR below its MA
            atr_fail = self.atr[-1] < self.atr_ma[-1]
            
            if exit_signal or atr_fail:
                self.position.close()
                reason = "VI Reversal" if exit_signal else "ATR Contraction"
                print(f"🌙✨🌑 MOON DEV EXIT: Closing at {current_close:.2f} | Reason: {reason}")

# Data Handling 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Map to backtesting.py format
data = data.rename(columns={
    '