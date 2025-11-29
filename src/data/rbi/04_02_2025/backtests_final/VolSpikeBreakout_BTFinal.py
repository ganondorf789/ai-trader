Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib

# Load and preprocess data with Moon Dev precision 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names with cosmic care ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolSpikeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (fixed as fraction)
    
    def init(self):
        # Bollinger Bands Upper (20,2) with Moon Dev precision 🌕
        def bb_upper(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
            upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, 
                                      nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            return upper
        
        self.upper_bb = self.I(bb_upper, self.data.Close, 
                              timeperiod=20, nbdevup=2, name='UPPER_BB')
        
        # 20-period SMA for exit (cosmic smooth) 🌌
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20, name='SMA_20')
        
        # 20-period average volume (lunar gravity measurement) 🌑
        self.avg_volume = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='AVG_VOL')

    def next(self):
        if len(self.data) < 20:
            return  # Skip early bars (not enough cosmic data) 🚀
        
        # Moon Dev debug prints (every 100 bars) 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 MOON DEV DEBUG | Bar: {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f} | Upper BB: {self.upper_bb[-1]:.2f} | SMA: {self.sma[-1]:.2f} | Volume: {self.data.Volume[-1]:.2f}")

        # Entry logic (cosmic alignment) ✨
        if not self.position:
            if len(self.data) < 2:
                return
            
            # Check previous bar's conditions (lunar phase analysis) 🌓
            prev_close = self.data.Close[-2]
            prev_upper = self.upper_bb[-2]
            prev_vol = self.data.Volume[-2]
            prev_avg_vol = self.avg_volume[-2]

            if prev_close > prev_upper and prev_vol >= 2 * prev_avg_vol:
                entry_price = self.data.Open[-1]
                sl_price = self.data.Low[-2]
                
                risk_per_share = entry_price - sl_price
                if risk_per_share <= 0:
                    return  # Avoid black hole trades 🌑
                
                # Position sizing with cosmic rounding (whole units only) 🪐
                position_size = (self.risk_per_trade * self.equity) / risk_per_share
                position_size = int(round(position_size))  # Ensure whole number of units
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🌙✨🚀 MOON DEV ENTRY SIGNAL | Size: {position_size} units | Entry: {entry_price:.2f} | SL: {sl_price:.2f} | Equity: {self.equity:.2f}")

        # Exit logic (when stars align) 🌟
        else:
            if self.data.Low[-1] <= self.sma[-1]:
                self.sell()
                print(f"🌙✨🌑 MOON DEV EXIT SIGNAL | Price: {self.data.Close[-1]:.2f} | SMA: {self.sma[-1]:.2f} | PnL: {self.position.pl:.2f}")

# Run backtest with Moon Dev cosmic parameters 🌕