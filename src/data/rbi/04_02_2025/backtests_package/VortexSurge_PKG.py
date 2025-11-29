I'll fix the code by removing all `backtesting.lib` imports and replacing the crossover functions with proper implementations. Here's the corrected version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = prepare_data(data_path)

class VortexSurge(Strategy):
    risk_pct = 0.01
    atr_multiplier = 1.5
    volume_threshold = 1.5
    volume_period = 20
    atr_period = 14
    atr_avg_period = 30
    max_bars_held = 5

    def init(self):
        # Vortex Indicator 🌪️
        high, low, close = self.data.High, self.data.Low, self.data.Close
        vortex = ta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda: vortex['VORTl_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTs_14'], name='VI-')

        # Volume Surge 📈
        self.vol_sma = self.I(talib.SMA, self.data.Volume, self.volume_period, name='Vol SMA')
        
        # ATR Expansion 📏
        self.atr14 = self.I(talib.ATR, high, low, close, self.atr_period, name='ATR14')
        self.atr30_avg = self.I(talib.SMA, self.atr14, self.atr_avg_period, name='ATR30 Avg')

        self.entry_bar = None

    def next(self):
        price = self.data.Close[-1]
        
        # Moon Dev Status Update 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} | Price: {price:.2f} | Equity: {self.equity:,.2f}")

        if not self.position:
            # Entry Conditions Check ✅
            vi_plus_cross = (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1])
            if (vi_plus_cross and
                self.data.Volume[-1] >= self.volume_threshold * self.vol_sma[-1] and
                self.atr14[-1] > self.atr30_avg[-1]):
                
                # Risk Management 💼
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr14[-1]
                stop_loss = price - self.atr_multiplier * atr_value
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.entry_bar = len(self.data)
                        self.buy(size=position_size, sl=stop_loss, tag="Moon Surge Entry 🚀")
                        print(f"\n🌙✨🚀 ENTRY SIGNAL @ {price:.2f}")
                        print(f"Size: {position_size} | SL: {stop_loss:.2f} | Risk: {self.risk_pct*100}%")

        else:
            # Exit Conditions 🚪
            current_bar = len(self.data)
            bars_held = current_bar - self.entry_bar
            
            vi_minus_cross = (self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1])
            if vi_minus_cross:
                self.position.close()
                print(f"\n🌙⏹️ VI EXIT @ {price:.2f} | Bars