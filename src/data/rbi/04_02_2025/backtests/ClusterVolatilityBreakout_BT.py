To address the problem, here is the implementation of the ClusterVolatilityBreakout strategy using backtesting.py:

```python
# 🌙 Moon Dev's ClusterVolatilityBreakout Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Data preparation function
def prepare_data(data_path):
    # Load and clean data
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Ensure proper column names
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=column_map)
    
    # Convert datetime and set index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

class ClusterVolatilityBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    daily_max_loss = 0.05  # 5% daily loss limit
    
    def init(self):
        # 🌙 Indicators Setup
        # Bollinger Bands with bandwidth calculation
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', which=2)
        self.bb_mid = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_MID', which=1)
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid, name='BB_WIDTH')
        
        # Volatility and volume indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA')
        
        # 🌙 Daily loss tracking
        self.current_day = None
        self.daily_start_equity = None
        self.trading_halted = False

    def next(self):
        # 🌙 Daily risk check
        current_time = self.data.index[-1]
        current_day = current_time.date()
        
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_start_equity = self.equity
            self.trading_halted = False
            print(f"\n🌙 New Trading Day: {current_day} | Starting Equity: ${self.daily_start_equity:,.2f}")
        
        if self.trading_halted:
            return

        # Check daily loss limit
        if self.daily_start_equity:
            daily_pnl = self.equity - self.daily_start_equity
            loss_pct = abs(daily_pnl) / self.daily_start_equity if daily_pnl < 0 else 0
            if loss_pct >= self.daily_max_loss:
                self.position.close()
                self.trading_halted = True
                print(f"🚨🌙 Daily Loss Limit Triggered! Stopping trading for {current_day}")
                return

        # 🌙 Strategy Logic
        if self.position:
            # Emergency exit check
            if self.bb_width[-1] > 0.03:
                self.position.close()
                print(f"🚨🌙 Emergency Exit! Bandwidth {self.bb_width[-1]:.2%} > 3%")
            return

        # Entry conditions
        price = self.data.Close[-1]
        cluster_high = self.data['liquidation_high'][-1]  # Assuming column exists
        cluster_low = self.data['liquidation_low'][-1]    # Assuming column exists
        
        # Long entry
        if (price > cluster_high and
            self.bb_width[-1] < 0.01 and
            self