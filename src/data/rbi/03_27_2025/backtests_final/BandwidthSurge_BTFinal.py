Here's the fully debugged and Moon Dev themed backtest code for the BandwidthSurge strategy:

```python
# 🌙 Moon Dev's BandwidthSurge Backtest Implementation ✨
from backtesting import Strategy, Backtest
import pandas as pd
import talib
import numpy as np

# Data Preparation 🌙
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names with lunar precision
    data.columns = data.columns.str.strip().str.lower()
    # Remove any cosmic interference (unnamed columns)
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Align columns with celestial coordinates
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    print("🌙✨ Data loaded and purified with lunar energy! Columns aligned with celestial coordinates.")
    return data

# Strategy Implementation 🚀
class BandwidthSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade (moon-approved risk level)
    
    def init(self):
        # Bollinger Band Indicators 🌗
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close)
        self.bb_low = self.I(talib.MIN, self.bb_width, 10)
        
        # Volume Confirmation 📈
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Volatility Management 🌊
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_sma = self.I(talib.SMA, self.atr, 20)
        
        print("🌙✨ Strategy Initialized with Moon Power! All indicators aligned with lunar cycles!")

    def next(self):
        # Skip early bars (waiting for full moon phase)
        if len(self.data) < 40:
            return
        
        # Entry Conditions Check 🌌
        price = self.data.Close[-1]
        entry_conditions = (
            self.bb_width[-1] == self.bb_low[-1],
            self.data.Volume[-1] > 2*self.vol_sma[-1],
            self.atr[-1] > self.atr_sma[-1]
        )
        
        # Moon Entry Logic 🌕
        if all(entry_conditions) and not self.position:
            risk_amount = self.equity * self.risk_percent
            atr_value = self.atr[-1]
            
            if atr_value == 0:
                print("🌙⚠️ Zero ATR detected - skipping trade to avoid division by zero!")
                return
                
            # Calculate position size with lunar precision (whole units only)
            position_size = int(round(risk_amount / (2 * atr_value)))
            
            if position_size > 0:
                self.buy(size=position_size)
                print(f"🌙🚀 LONG Activated by lunar gravitational pull! Size: {position_size} units @ {price:.2f}")
                # Initialize trailing stop with moon's protection
                for trade in self.trades:
                    trade.trail_target = 0  # Will track highest high
        
        # Adaptive Exit Management 🌗
        for trade in self.trades:
            if trade.is_long:
                # Update trailing stop with moon's guidance
                high_since_entry = self.data.High[trade.entry_bar:]
                current_peak = high_since_entry.max()
                trail_stop = current_peak - 2*self.atr[-1]
                
                # Update only if higher than previous stop
                if trail_stop > trade.sl or pd.isnull(trade.sl):
                    trade.sl = trail_stop
                    print(f"🌙🛑 Trail Stop Updated by moon's protective shield: {trail_stop:.2f}")

# Backtest Execution 🌙
if __name__ == "__main__":
    print("🌑🌒🌓🌔🌕🌖🌗🌘 Moon Dev's BandwidthSurge Strategy Activated 🌑🌒🌓🌔🌕🌖🌗🌘")
    data = load_data('/Users/md/Dropbox/dev/github/moon