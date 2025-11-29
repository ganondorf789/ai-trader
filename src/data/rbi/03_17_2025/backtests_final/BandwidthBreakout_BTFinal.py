I'll help debug and fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints and proper position sizing:

```python
from backtesting import Strategy, Backtest
import pandas as pd
import talib

# Load and preprocess data 🌙
print("🌌 Initializing Moon Dev Data Systems...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("🌕 Data successfully loaded and preprocessed!")

class BandwidthBreakout(Strategy):
    def init(self):
        # Moon-powered indicators ✨
        print("🌙 Initializing Moon Dev Indicators...")
        self.upper_band, self.middle_band, self.lower_band = self.I(talib.BBANDS, 
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Calculate Bollinger Band Width 🌗
        self.bb_width = (self.upper_band - self.lower_band) / self.middle_band
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=10)
        
        # Volume analysis 📊
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, timeperiod=50)
        
        # ATR for risk management 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.trailing_stop = None
        print("✨ Indicators successfully initialized!")

    def notify_trade(self, trade):
        if trade.is_open:
            # Set initial trailing stop 🌙
            atr_value = self.atr[-1]
            if trade.is_long:
                self.trailing_stop = trade.entry_price - 2 * atr_value
                print(f"🚀🌙 LONG LAUNCH! Entry: {trade.entry_price:.2f} | Initial Stop: {self.trailing_stop:.2f}")
            else:
                self.trailing_stop = trade.entry_price + 2 * atr_value
                print(f"🌙🚀 SHORT BLAST! Entry: {trade.entry_price:.2f} | Initial Stop: {self.trailing_stop:.2f}")

    def next(self):
        # Wait for indicators to stabilize 🌙
        if len(self.data) < 50:
            return

        # Entry constellation alignment 🌌
        if not self.position:
            bb_cond = self.bb_width[-1] <= self.bb_width_low[-1]
            vol_cond = self.data.Volume[-1] > self.volume_sma50[-1]
            
            if bb_cond and vol_cond:
                # Long signal 🌕
                if self.data.Close[-1] > self.upper_band[-1]:
                    risk = self.equity * 0.01
                    stop_dist = 2 * self.atr[-1]
                    size = int(round(risk / stop_dist))  # Round to whole units
                    if size > 0:
                        self.buy(size=size)
                        print(f"🌕🚀 MOON SHOT! LONG {size} units | ATR: {stop_dist:.2f}")
                
                # Short signal 🌑
                elif self.data.Close[-1] < self.lower_band[-1]:
                    risk = self.equity * 0.01
                    stop_dist = 2 * self.atr[-1]
                    size = int(round(risk / stop_dist))  # Round to whole units
                    if size > 0:
                        self.sell(size=size)
                        print(f"🚀🌑 DARK SIDE! SHORT {size} units | ATR: {stop_dist:.2f}")

        # Trailing stop management 🌙
        else:
            if self.position.is_long:
                new_stop = self.data.High[-1] - 2 * self.atr[-1]
                self.trailing_stop = max(self.trailing_stop, new_stop)