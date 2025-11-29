```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Clean and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and handle data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySpike(Strategy):
    short_term_period = 5  # 5-period ATR (15m*5 = 75min)
    daily_period = 96      # 96-period ATR (15m*96 = 24hr)
    volume_ma_period = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators with TA-Lib
        self.short_term_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.short_term_period, name='ATR_5m')
        self.daily_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.daily_period, name='ATR_1D')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='VOL_MA20')
        
    def next(self):
        # Moon Dev's Cosmic Entry Check 🌌
        if not self.position:
            if (len(self.short_term_atr) > 1 and len(self.daily_atr) > 1 and
                self.short_term_atr[-1] > 1.5 * self.daily_atr[-1] and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                # Calculate position size with stellar precision 🌠
                risk_amount = self.equity * self.risk_pct
                entry_atr = self.short_term_atr[-1]
                stop_distance = 2 * entry_atr
                
                if stop_distance <= 0:
                    print("🌙 MOON DEV WARNING: Zero stop distance - aborting launch!")
                    return
                
                position_size = int(round(risk_amount / stop_distance))
                if position_size == 0:
                    print(f"🌙 MOON DEV WARNING: Zero position size at {self.data.index[-1]}")
                    return
                
                # Execute FTL jump to LONG position 🚀
                self.buy(size=position_size)
                self.entry_atr = entry_atr  # Store for exit calculations
                self.highest_high = self.data.High[-1]  # Initialize trailing stop
                
                # Print launch sequence initiated
                print(f"\n🌙✨ MOON DEV TRADE LAUNCH! ✨🌙")
                print(f"   🚀 Entry Price: {self.data.Open[-1]:.2f}")
                print(f"   📈 ATR Ratio: {self.short_term_atr[-1]/self.daily_atr[-1]:.2f}x")
                print(f"   💰 Position Size: {position_size} units")
                print(f"   🛡️  Initial Stop: {self.data.Open[-1] - stop_distance:.2f}")
        
        # Galactic Trailing Stop Management 🌠
        else:
            # Update cosmic high watermark
            self.highest_high = max(self.highest_high, self.data.High[-1])
            trailing_stop = self.highest_high - 2 * self.entry_atr
            
            # Check for gravitational pull exit 🌑
            if self.data.Close[-1] < trailing_stop:
                self.position.close()
                print(f"\n🌙🌑 MOON DEV GRAVITY EXIT! 🌑🌙")
                print(f"   💸 Exit Price: {self.data.Close[-1]:.2f}")
                print(f"   🏆 Peak Altitude: {self.highest_high:.2f}")
                print(f"   📉 Final Stop Level: {trailing_stop:.2f}")

# Run backtest with 1M cosmic credits 🌕
bt = Backtest(data, VolatilitySpike,