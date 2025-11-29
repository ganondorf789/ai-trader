Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed enhancements:

```python
# 🌙 Moon Dev Backtest AI Implementation: VolumetricBreakout Strategy 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ======== DATA PREPARATION ========
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ======== STRATEGY CLASS ========
class VolumetricBreakout(Strategy):
    def init(self):
        # 🌙✨ BOLLINGER BAND CALCULATIONS
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20, name='SMA20')
        self.std20 = self.I(talib.STDDEV, self.data.Close, timeperiod=20, nbdev=1, name='STD20')
        self.bb_upper = self.I(lambda: self.sma20 + 2.5*self.std20, name='BB_Upper')
        self.bb_lower = self.I(lambda: self.sma20 - 2.5*self.std20, name='BB_Lower')

        # 🌊 VOLUME THRESHOLD CALCULATIONS
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Vol_SMA20')
        self.volume_threshold = self.I(lambda: self.volume_sma20 * 1.5, name='Vol_Threshold')

        print("🌙✨🚀 Moon Dev Indicators Initialized! Ready for cosmic profits! 🌙✨🚀")

    def next(self):
        # Skip first 20 bars for warmup
        if len(self.data) < 20:
            return

        # 🌙 REAL-TIME VALUES
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        vol_thresh = self.volume_threshold[-1]

        # 💰 RISK MANAGEMENT
        risk_pct = 0.02  # 2% risk per trade

        # ======== ENTRY LOGIC ========
        if not self.position:
            # 🚀 LONG ENTRY: Price above BB_Upper + Volume Spike
            if price > upper and volume > vol_thresh:
                risk_amount = self.equity * risk_pct
                stop_loss = self.sma20[-1]
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.entry_bar = len(self.data)
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🚀🌕 MOON SHOT! LONG {position_size} @ {price:.2f} | Vol {volume:.2f} > {vol_thresh:.2f} 🌕🚀")

            # 🌑 SHORT ENTRY: Price below BB_Lower + Volume Spike
            elif price < lower and volume > vol_thresh:
                risk_amount = self.equity * risk_pct
                stop_loss = self.sma20[-1]
                risk_per_share = stop_loss - price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.entry_bar = len(self.data)
                        self.sell(size=position_size, sl=stop_loss)
                        print(f"🌑🌙 BLACK HOLE! SHORT {position_size} @ {price:.2f} | Vol {volume:.2f} > {vol_thresh:.2f} 🌙🌑")

        # ======== EXIT LOGIC ========
        else:
            # 💫 EXIT CONDITIONS
            bars_held = len(self.data) - self.entry_bar
            if self.position.is_long and (price < upper or bars_held >= 5):