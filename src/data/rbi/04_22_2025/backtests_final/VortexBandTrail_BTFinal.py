```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as pta

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Calculate 4h ATR
data_4h = data.resample('4H').agg({
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
})
atr_4h = talib.ATR(data_4h['High'], data_4h['Low'], data_4h['Close'], timeperiod=14)
data['ATR_4H'] = atr_4h.reindex(data.index, method='ffill')

class VortexBandTrail(Strategy):
    def init(self):
        # Indicators
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        vi = pta.vortex(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close, 
            length=14
        )
        self.vi_plus = self.I(lambda: vi['VIn_14'], name='VI+')
        self.vi_minus = self.I(lambda: vi['VIm_14'], name='VI-')
        
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr_4h = self.I(lambda: self.data['ATR_4H'], name='ATR_4H')
        
        # Risk management
        self.consecutive_losses = 0
        self.open_positions = 0
        self.trailing_stop = None
        self.prev_pos_size = 0

    def next(self):
        # Track position changes
        current_pos = self.position.size if self.position else 0
        if self.prev_pos_size != 0 and current_pos == 0:
            if self.trades[-1].pl < 0:
                self.consecutive_losses += 1
                print(f"🌧️ Consecutive Losses: {self.consecutive_losses} ✨")
            else:
                self.consecutive_losses = 0
                print("🌈 Profit! Reset consecutive losses 🌙")
        self.prev_pos_size = current_pos

        if self.position:
            # Update trailing stop
            new_stop = self.data.High[-1] - 2 * self.atr_4h[-1]
            if new_stop > self.trailing_stop or not self.trailing_stop:
                self.trailing_stop = new_stop
                print(f"🌙 Trail Stop Updated: {self.trailing_stop:.2f} 🛑")

            # Exit conditions
            if self.data.Low[-1] <= self.trailing_stop:
                print(f"🚨 STOP LOSS @ {self.trailing_stop:.2f} 🩸")
                self.position.close()
                self.open_positions -= 1
            elif (self.vi_minus[-2] > self.vi_plus[-2] and self.vi_minus[-1] < self.vi_plus[-1]):
                print(f"⚠️ VI CROSS EXIT @ {self.data.Close[-1]:.2f} ⏳")
                self.position.close()
                self.open_positions -= 1
            elif self.data.High[-1] >= self.bb_upper[-1]:
                print(f"🎯 UPPER BAND EXIT @ {self.data.Close[-1]:.2f} 💰")
                self.position.close()
                self.open_positions -= 1
        else:
            # Entry conditions
            price_condition = self.data.Low[-1] <= self.bb_lower[-1]
            volume_condition = self.data.Volume[-1] > self.volume_sma[-1]
            vi_condition = (self.vi_plus[-1] > self.vi_minus[-1]) and (