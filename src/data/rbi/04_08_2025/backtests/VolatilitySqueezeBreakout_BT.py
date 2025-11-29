```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert index to datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeBreakout(Strategy):
    def init(self):
        # Calculate indicators with TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.I(lambda: bb_upper, name='BB_upper')
        self.I(lambda: bb_lower, name='BB_lower')

        # Keltner Channel
        ema20 = talib.EMA(close, timeperiod=20)
        atr20 = talib.ATR(high, low, close, timeperiod=20)
        kc_upper = ema20 + 1.5 * atr20
        kc_lower = ema20 - 1.5 * atr20
        self.I(lambda: kc_upper, name='KC_upper')
        self.I(lambda: kc_lower, name='KC_lower')

        # Volume SMA50
        volume_sma50 = talib.SMA(volume, timeperiod=50)
        self.I(lambda: volume_sma50, name='Volume_SMA50')

        # ATR14 for trailing stop
        atr14 = talib.ATR(high, low, close, timeperiod=14)
        self.I(lambda: atr14, name='ATR14')

        # Strategy variables
        self.squeeze_count = 0
        self.squeeze_confirmed = False
        self.consecutive_losses = 0

    def next(self):
        if self.position:
            atr14 = self.I('ATR14')[-1]
            if self.position.is_long:
                # Update trailing high
                self.trailing_high = max(self.trailing_high, self.data.High[-1])
                trailing_stop = self.trailing_high - 2 * atr14
                if self.data.Low[-1] <= trailing_stop:
                    print(f"🌙🛑 LONG Trailing Stop Hit at {trailing_stop:.2f}")
                    self.position.close()
                else:
                    # Check squeeze re-entry
                    if (self.I('BB_upper')[-1] < self.I('KC_upper')[-1]) and \
                       (self.I('BB_lower')[-1] > self.I('KC_lower')[-1]):
                        print(f"🌙⚠️ Emergency Exit: Squeeze Re-entered")
                        self.position.close()
            else:
                # Update trailing low
                self.trailing_low = min(self.trailing_low, self.data.Low[-1])
                trailing_stop = self.trailing_low + 2 * atr14
                if self.data.High[-1] >= trailing_stop:
                    print(f"🌙🛑 SHORT Trailing Stop Hit at {trailing_stop:.2f}")
                    self.position.close()
                else:
                    if (self.I('BB_upper')[-1] < self.I('KC_upper')[-1]) and \
                       (self.I('BB_lower')[-1] > self.I('KC_lower')[-1]):
                        print(f"🌙⚠️ Emergency Exit: Squeeze Re-entered")
                        self.position.close()
            return

        # Check consecutive losses
        if self.consecutive_losses >= 3:
            print("🌙🔴 3 Consecutive Losses - Trading Halted")
            return

        # Check squeeze condition
        current_bb_upper = self.I('BB_upper')[-1]
        current_bb_lower = self.I('BB_lower')[-1]
        current_kc_upper = self.I('KC_