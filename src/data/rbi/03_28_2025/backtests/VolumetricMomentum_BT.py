```python
# 🌙 Moon Dev's VolumetricMomentum Backtest ✨
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import talib

# Data preparation 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare cosmic data ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index(pd.to_datetime(data['datetime']), inplace=True)

class VolumetricMomentum(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    stop_pct = 0.02   # 2% initial stop
    rr_ratio = 3      # Cosmic reward ratio 🚀
    swing_period = 20 # Swing low lookback
    
    def init(self):
        # 🌟 Celestial Indicators Calculation
        periods_per_day = 96  # 15m intervals
        
        # VWAP Cross (5-day vs 50-day)
        self.vwap5 = self.I(ta.vwma, self.data.Close, self.data.Volume, length=5*periods_per_day)
        self.vwap50 = self.I(ta.vwma, self.data.Close, self.data.Volume, length=50*periods_per_day)
        
        # Chande Momentum Oscillator (14-day)
        self.cmo = self.I(ta.cmo, self.data.Close, length=14*periods_per_day)
        
        # FRAMA Exit (10-day)
        self.frama = self.I(ta.frama, self.data.Close, length=10*periods_per_day)
        
        # Swing Low for stop calculation
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
    def next(self):
        # Wait for cosmic alignment of data ✨
        if len(self.data) < 50*96:  # Wait for full VWAP50 calculation
            return
        
        price = self.data.Close[-1]
        
        # 🌙 Entry Constellation
        if not self.position:
            # VWAP crossover and momentum confirmation
            if crossover(self.vwap5, self.vwap50) and self.cmo[-1] > 50:
                # Starlight stop calculation ✨
                entry_price = price
                fixed_stop = entry_price * (1 - self.stop_pct)
                swing_stop = self.swing_low[-1]
                vwap_stop = self.vwap50[-1]
                
                stop_price = max(fixed_stop, swing_stop, vwap_stop)
                risk_per_unit = entry_price - stop_price
                
                if risk_per_unit > 0:
                    # Calculate cosmic position size 🌌
                    risk_amount = self.risk_pct * self.equity
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size > 0:
                        take_profit = entry_price + (risk_per_unit * self.rr_ratio)
                        self.buy(size=position_size, 
                                sl=stop_price,
                                tp=take_profit)
                        print(f"🌙✨🚀 LIFT OFF! Buying {position_size} @ {entry_price:.2f}")
                        print(f"   🛡️ Cosmic Shield: {stop_price:.2f} | 🌠 Profit Star: {take_profit:.2f}")
        
        # 🛑 Exit Black Hole Detection
        else:
            # FRAMA exit condition (2 consecutive closes below)
            if len(self.data.Close) > 1:
                current_below = self.data.Close[-1] < self.frama[-1]
                prev_below = self.data.Close[-2] < self.frama[-2]
                
                if current_below and prev_below:
                    self.position.close()
                    print(f"🌙🌧️ STARDUST FALL! Exiting @ {price:.2f}")

# Launch backtest rocket 🚀
bt = Backtest(data, VolumetricMomentum, cash=1_000_000, exclusive_orders=True)
stats = bt.run()

# Print cosmic performance report 🌌
print("\n🌙✨ FINAL MISSION REPORT:")
print(stats)
print("\n🌠 STRATEG