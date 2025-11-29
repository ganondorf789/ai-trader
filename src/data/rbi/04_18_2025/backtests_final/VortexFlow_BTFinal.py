Here's the debugged version of your backtest code with Moon Dev themed improvements 🌙✨:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation - Moon Dev style path handling 🌙
try:
    data = pd.read_csv('BTC-USD-15m.csv')  # Simplified path for portability
except FileNotFoundError:
    print("🌑 MOON ALERT: Data file not found! Please check the path.")
    raise

# Clean and format data with Moon Dev precision ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

try:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
except KeyError:
    print("🌒 MOON ALERT: 'datetime' column missing! Check your data format.")
    raise

class VortexFlow(Strategy):
    def init(self):
        # Calculate indicators with self.I() wrapper
        high, low, close = self.data.High, self.data.Low, self.data.Close
        
        # Vortex Indicator with Moon Dev validation 🌕
        try:
            vi = ta.vortex(high=high, low=low, close=close, length=14)
            self.vi_plus = self.I(lambda: vi['VORTEX_14U'], name='VI+')
            self.vi_minus = self.I(lambda: vi['VORTEX_14D'], name='VI-')
        except Exception as e:
            print(f"🌓 MOON ALERT: Vortex Indicator error - {str(e)}")
            raise
        
        # MFI and Volume MA with proper error handling 🌔
        try:
            self.mfi = self.I(talib.MFI, high, low, close, self.data.Volume, timeperiod=14)
            self.volume_ma5 = self.I(talib.SMA, self.data.Volume, timeperiod=5)
        except Exception as e:
            print(f"🌔 MOON ALERT: Indicator calculation error - {str(e)}")
            raise
        
        # Trend filter and risk management 🌖
        try:
            self.ma200 = self.I(talib.SMA, close, timeperiod=200)
            self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        except Exception as e:
            print(f"🌖 MOON ALERT: Trend filter error - {str(e)}")
            raise

    def next(self):
        # Moon Dev debug prints 🌙
        if len(self.data.Close) < 2:
            return

        current_close = self.data.Close[-1]
        moon_emoji = '🌑🌒🌓🌔🌕🌖🌗🌘'[len(self.data) % 8]

        # Entry logic with proper position sizing 🌕
        if not self.position:
            vi_bullish = (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
            mfi_bullish = (self.mfi[-2] < 30) and (self.mfi[-1] > 30)
            volume_spike = self.data.Volume[-1] > self.volume_ma5[-1]
            trend_up = current_close > self.ma200[-1]

            if all([vi_bullish, mfi_bullish, volume_spike, trend_up]):
                risk_amount = self.equity * 0.01  # 1% risk per trade
                atr_value = self.atr[-1]
                stop_price = current_close - 2 * atr_value
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:  # Moon Dev safety check 🌙
                        self.buy(size=size, sl=stop_price)
                        print(f"{moon_emoji}✨🚀 MOON ENTRY: {current_close:.2f} | Size: {size} | Risk: {2*atr_value:.2f}")
                    else:
                        print(f"{moon_emoji}🌑 MOON WARNING: Zero size calculated!")

        # Exit logic with proper position handling 🌗