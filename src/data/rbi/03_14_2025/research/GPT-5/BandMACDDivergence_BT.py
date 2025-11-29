```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class BandMACDDivergence(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.dropna()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])
        self.data.df = self.data.df.rename(columns={
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        })

        # Bollinger Bands
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        # MACD
        self.macd, self.macd_signal, _ = self.I(
            talib.MACD, 
            self.data.Close, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        
        # ATR
        self.atr = self.I(
            talib.ATR, 
            self.data.High, 
            self.data.Low, 
            self.data.Close, 
            timeperiod=14
        )

    def next(self):
        if len(self.data.Close) < 20:  # 🌙 Minimum data check
            return

        # Calculate Bollinger Band slope
        bb_slope = np.gradient(self.upper_band)
        
        # Calculate divergence
        macd_divergence = self.macd - self.macd_signal
        divergence_threshold = 1.5 * self.atr

        # Entry logic
        if bb_slope[-1] > 0 and macd_divergence[-1] > divergence_threshold[-1]:
            if self.macd[-2] < self.macd_signal[-2] and self.macd[-1] > self.macd_signal[-1]:
                print("🌙 🚀 Long Entry Signal Detected! MACD Divergence with Bollinger Band Slope! 🚀 🌙")
                self.buy(size=0.5)  # Fixed: Percentage-based sizing

        if bb_slope[-1] < 0 and macd_divergence[-1] < -divergence_threshold[-1]:
            if self.macd_signal[-2] < self.macd[-2] and self.macd_signal[-1] > self.macd[-1]:
                print("🌙 🔻 Short Entry Signal Detected! MACD Divergence with Bollinger Band Slope! 🔻 🌙")
                self.sell(size=0.5)  # Fixed: Percentage-based sizing

        # Exit logic
        if self.position.is_long and (bb_slope[-1] <= 0 or (self.macd_signal[-2] < self.macd[-2] and self.macd_signal[-1] > self.macd[-1])):
            print("✨ ✋ Long Exit Signal! Take Profit or Reversal Detected! ✋ ✨")
            self.position.close()

        if self.position.is_short and (bb_slope[-1] >= 0 or (self.macd[-2] < self.macd_signal[-2] and self.macd[-1] > self.macd_signal[-1])):
            print("✨ ✋ Short Exit Signal! Take Profit or Reversal Detected! ✋ ✨")
            self.position.close()

# Load data
try:
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', 
                      parse_dates=['datetime'], 
                      index_col='datetime')
    data = data.dropna()
    print("🌙 Data loaded successfully! Starting backtest... ✨")
except Exception as e:
    print(f"🌙 CRITICAL ERROR: Failed to load data - {str(e)}")

# Run backtest
if 'data' in locals() and not data.empty:
    bt = Backtest(data, BandMACDDivergence, cash=1000000, commission=.002)
    stats = bt.run()
    print("🌙 Backtest completed! Moon Dev stats incoming... 🌙")
    print