```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Define BBANDS helper function
        def get_bb(close, band_type, timeperiod=20, nbdevup=2, nbdevdn=2):
            upper, middle, lower = talib.BBANDS(close, 
                                               timeperiod=timeperiod,
                                               nbdevup=nbdevup,
                                               nbdevdn=nbdevdn)
            return {'upper': upper, 'middle': middle, 'lower': lower}[band_type]
        
        # Calculate indicators
        self.upper = self.I(get_bb, self.data.Close, 'upper', 20, 2, 2)
        self.middle = self.I(get_bb, self.data.Close, 'middle', 20, 2, 2)
        self.lower = self.I(get_bb, self.data.Close, 'lower', 20, 2, 2)
        
        # Calculate BB Width
        bb_width = (self.upper - self.lower) / self.middle
        
        # BB Width Percentile (100-period lookback)
        def calc_percentile(series):
            return series.expanding().apply(lambda x: (x[-1] <= x[-100:]).mean()*100)
        
        self.bb_width_pct = self.I(calc_percentile, bb_width)
        
        # ADX (14-period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Volume MA (20-period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20)
        
    def next(self):
        # Moon Dev debug prints 🌙
        if len(self.data) % 500 == 0:
            print(f"\n🌙 MOON DEV DEBUG [{self.data.index[-1]}]")
            print(f"✨ BB Width %: {self.bb_width_pct[-1]:.1f} | ADX: {self.adx[-1]:.1f}")
            print(f"🚀 Volume: {self.data.Volume[-1]:.0f} vs {self.vol_ma[-1]:.0f}")
        
        # Entry logic
        if not self.position:
            # Common conditions
            squeeze_condition = self.bb_width_pct[-1] < 10
            adx_rising = self.adx[-1] > self.adx[-2] if len(self.adx) > 1 else False
            adx_strong = self.adx[-1] > 25
            vol_ok = self.data.Volume[-1] > self.vol_ma[-1]
            
            # Long entry
            if all([squeeze_condition, adx_rising, adx_strong, vol_ok,
                   self.data.Close[-1] > self.upper[-1]]):
                
                sl_price = self.middle[-1]
                risk = self.data.Close[-1] - sl_price
                position_size = int(round((self.equity * self.risk_percent) / risk))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=self.data.Close[-1] + 2*risk)
                    print(f"\n🌕🌕🌕 MOON DEV LONG SIGNAL 🌕🌕🌕")
                    print(f"Entry: {self.data.Close[-1]:.2f}")
                    print(f"Size: {position_size} contracts")
                    print(f"Risk: {risk:.2f} | R/R: 1:2")
            
            # Short entry
            elif all([squeeze_condition, adx_