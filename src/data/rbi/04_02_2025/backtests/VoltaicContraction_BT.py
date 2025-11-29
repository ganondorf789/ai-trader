```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicContraction(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Calculate Indicators
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_upper')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_lower')
        
        # Keltner Channels
        self.kc_middle = self.I(talib.EMA, self.data.Close, 20, name='KC Middle')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR')
        self.kc_upper = self.I(lambda: self.data['KC Middle'] + 2 * self.data['ATR'], name='KC Upper')
        self.kc_lower = self.I(lambda: self.data['KC Middle'] - 2 * self.data['ATR'], name='KC Lower')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume MA')
        
        # Elder Impulse System
        self.ema13 = self.I(talib.EMA, self.data.Close, 13, name='EMA13')
        _, _, self.macd_hist = self.I(talib.MACD, self.data.Close, 12, 26, 9, name='MACD')
        
        # Trailing ATR
        self.atr_trailing = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.contraction_count = 0
        print("✨ Moon Dev Indicators Initialized! Ready for cosmic analysis 🚀")

    def next(self):
        # 🌙 Check existing positions
        if self.position:
            if self.position.is_long:
                # Long exit conditions
                if self.data.Close[-1] < self.kc_lower[-1]:
                    print(f"🌙 Closing LONG at {self.data.Close[-1]} (KC Lower Break)")
                    self.position.close()
                elif self.eis_current == -1:
                    print(f"🌙 Closing LONG at {self.data.Close[-1]} (EIS Bearish Flip)")
                    self.position.close()
                else:
                    # Update trailing stop
                    trail = self.data.Close[-1] - 3 * self.atr_trailing[-1]
                    self.trailing_stop = max(trail, self.trailing_stop) if hasattr(self, 'trailing_stop') else trail
                    if self.data.Low[-1] < self.trailing_stop:
                        print(f"🚨 Trailing Stop Hit! Closing LONG at {self.trailing_stop}")
                        self.position.close()
            else:
                # Short exit conditions
                if self.data.Close[-1] > self.kc_upper[-1]:
                    print(f"🌙 Closing SHORT at {self.data.Close[-1]} (KC Upper Break)")
                    self.position.close()
                elif self.eis_current == 1:
                    print(f"🌙 Closing SHORT at {self.data.Close[-1]} (EIS Bullish Flip)")
                    self.position.close()
                else:
                    trail = self.data.Close[-1] + 3 * self.atr_trailing[-1]
                    self.trailing_stop = min