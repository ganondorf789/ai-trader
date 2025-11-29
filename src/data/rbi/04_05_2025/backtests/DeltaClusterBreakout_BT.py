```python
# 🌙 Moon Dev's DeltaClusterBreakout Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preprocessing
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Set datetime index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class DeltaClusterBreakout(Strategy):
    # Strategy parameters
    atr_period = 14
    hvn_window = 20
    risk_pct = 0.01
    trailing_multiplier = 1.5
    
    def init(self):
        # 🌙 Indicators using TA-Lib and proper self.I() wrapper
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, 20)
        
        # High-volume node detection
        self.vol_max = self.I(talib.MAX, self.data.Volume, self.hvn_window)
        
        # Delta proxy (Close vs Open)
        self.delta = self.I(lambda: self.data.Close - self.data.Open)
        
        # Swing highs/lows for price rejection
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        
        # Print Moon Dev initialization message
        print("🌙 Moon Dev Strategy Activated! 🚀\n")
        
    def next(self):
        # Skip initial bars for indicator warmup
        if len(self.data) < self.hvn_window + 5:
            return
            
        # 🌙 Moon Dev Core Logic
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_volume = self.data.Volume[-1]
        
        # Volatility contraction check
        volatility_ok = self.atr[-1] < self.atr_sma[-1]
        
        # High-volume node detection
        hvn_active = current_volume >= self.vol_max[-1]
        
        # Delta positive (basic proxy)
        delta_positive = self.delta[-1] > 0
        
        # Entry conditions
        if not self.position and volatility_ok and hvn_active and delta_positive:
            # Calculate risk parameters
            atr_value = self.atr[-1]
            stop_loss = current_high - (0.5 * atr_value)
            risk_per_share = current_close - stop_loss
            
            if risk_per_share > 0:
                # Moon Dev position sizing 🌙
                risk_amount = self.risk_pct * self.equity
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    # Enter long with initial stop loss
                    self.buy(size=position_size, sl=stop_loss, tag='DeltaClusterEntry')
                    print(f"🌙 MOON DEV ENTRY 🚀 | Price: {current_close:.2f} | Size: {position_size} | SL: {stop_loss:.2f} ✨")
        
        # 🌙 Dynamic trailing stops
        for trade in self.trades:
            if trade.is_long:
                # Calculate trailing stop
                high_since_entry = self.data.High[trade.entry_bar:len(self.data)].max()
                new_sl = high_since_entry - (self.trailing_multiplier * self.atr[-1])
                
                # Update stop loss if conditions met
                if new_sl > trade.sl:
                    trade.sl = max(new_sl, trade.sl)
                    print(f"🌙 TRAILING UPDATE 🌙 | New SL: {new_sl:.2f} | Current Price: {current_close:.2f}")

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
btc_data =