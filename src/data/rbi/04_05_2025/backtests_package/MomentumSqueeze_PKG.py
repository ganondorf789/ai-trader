Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper implementations, and Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns to match backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class MomentumSqueeze(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    stop_loss_pct = 0.03  # 3% trailing stop
    
    def init(self):
        # Core indicators
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14, name='CMO')
        
        # Bollinger Bands components
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_Lower')
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_Middle')
        
        # Trend filter
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA200')
        
        # Trailing price trackers
        self.highest_close = None
        self.lowest_close = None

    def next(self):
        # Skip early bars without indicator data
        if len(self.data) < 20 or len(self.cmo) < 14:
            return
        
        # Calculate Bollinger Width metrics
        current_bb_width = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_middle[-1]
        previous_bb_width = (self.bb_upper[-2] - self.bb_lower[-2])/self.bb_middle[-2]
        bb_contracting = current_bb_width < previous_bb_width
        
        # Calculate 20-period BB Width SMA
        all_bb_width = (self.bb_upper.array - self.bb_lower.array)/self.bb_middle.array
        bb_width_sma = talib.SMA(all_bb_width, timeperiod=20)[-1] if len(all_bb_width) >= 20 else None
        
        # Entry conditions
        if not self.position:
            # Long entry logic - bullish crossover replacement
            if ((self.cmo[-2] < 50 and self.cmo[-1] > 50) and  # Bullish crossover replacement
                bb_contracting and 
                self.data.Close[-1] > self.sma200[-1] and 
                (bb_width_sma is not None and current_bb_width < bb_width_sma)):
                
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]
                risk_per_share = entry_price * self.stop_loss_pct
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.highest_close = entry_price
                    self.buy(size=position_size)
                    print(f"🌙✨ BULLISH momentum confirmed! LONG {position_size} units @ {entry_price} 🚀")
                    print(f"🌕 Moon Dev AI detected optimal entry with CMO crossing 50 and BB contraction!")
            
            # Short entry logic - bearish crossover replacement
            elif ((self.cmo[-2]