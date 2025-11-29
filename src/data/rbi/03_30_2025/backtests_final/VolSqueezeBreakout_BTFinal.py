Here's the fully debugged code with Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data with Moon Dev cosmic energy 🌙
print("🌌 INITIATING MOON DEV DATA LOADING SEQUENCE...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data with lunar precision
print("✨ PURIFYING DATA WITH MOONLIGHT...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')

# Convert and set datetime index
print("⏳ ALIGNING TEMPORAL COORDINATES...")
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Standardize column names
print("📊 CALIBRATING DATA DIMENSIONS...")
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolSqueezeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        close = self.data.Close
        volume = self.data.Volume
        
        # Bollinger Bands (20,2)
        def bb_bands(close):
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            return upper, middle, lower
        
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            bb_bands, close, name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Bollinger Bandwidth
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            return (upper - lower) / middle
        
        self.bb_width = self.I(bb_width, close, name='BB_Width')
        
        # 20-period minimum bandwidth
        self.bb_width_min = self.I(
            lambda x: talib.MIN(x, timeperiod=20), self.bb_width, name='BB_Width_Min'
        )
        
        # Volume indicators
        self.volume_avg = self.I(
            talib.SMA, volume, timeperiod=20, name='Volume_Avg'
        )
        
        # Historical volatility (5-day stddev of returns)
        self.returns = self.I(talib.ROCP, close, timeperiod=1, name='Returns')
        self.hist_vol = self.I(
            talib.STDDEV, self.returns, timeperiod=5, nbdev=1, name='Hist_Vol'
        )
        self.avg_hist_vol = self.I(
            talib.SMA, self.hist_vol, timeperiod=20, name='Avg_Hist_Vol'
        )
        
        print("🌙 MOON DEV INDICATORS READY FOR LIFTOFF! 🚀")
        print("✨ ALL SYSTEMS GO - NO BACKTESTING.LIB DETECTED! ✨")

    def next(self):
        # Emergency volatility exit
        if self.position:
            current_vol = self.hist_vol[-1]
            avg_vol = self.avg_hist_vol[-1]
            if current_vol > 2 * avg_vol:
                self.position.close()
                print(f"🌪️ VOLATILITY STORM! Closing position at {self.data.Close[-1]} ✨")
        
        # Entry logic
        if not self.position:
            upper = self.bb_upper[-1]
            lower = self.bb_lower[-1]
            width_current = self.bb_width[-1]
            width_min = self.bb_width_min[-1]
            volume_current = self.data.Volume[-1]
            volume_avg = self.volume_avg[-1]
            
            # Long entry conditions
            if (self.data.High[-1] > upper and
                width_current <= width_min and
                volume_current >= 1.5 * volume_avg):
                
                sl_price = lower
                entry_price = self.data.Close[-1]
                risk_per_share = entry_price - sl_price
                
                if