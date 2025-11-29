Here's the complete implementation for the Volatility Squeeze Breakout strategy:

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySqueezeBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    hist_vol_period = 20
    atr_period = 14
    risk_pct = 0.01
    atr_multiplier = 1.5
    trailing_atr_multiplier = 2.0
    volatility_threshold = 500  # Adjust based on asset

    def init(self):
        close = self.data.Close
        
        # Calculate indicators with TA-Lib
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev
        )
        self.hist_vol = self.I(talib.STDDEV, close, timeperiod=self.hist_vol_period, name='HistVol')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, timeperiod=self.atr_period, name='ATR')
        
        # Track trade extremes
        self.trade_data = {}

    def next(self):
        if len(self.data) < self.bb_period:
            return

        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]
        hist_vol = self.hist_vol[-1]

        # Moon Dev Entry Logic 🌙✨
        if not self.position:
            if bb_width < hist_vol and current_atr < self.volatility_threshold:
                if current_close > self.bb_upper[-1]:
                    risk_amount = self.risk_pct * self.equity
                    stop_loss = current_close - (self.atr_multiplier * current_atr)
                    size = int(round(risk_amount / (self.atr_multiplier * current_atr)))
                    if size > 0:
                        trade = self.buy(size=size, sl=stop_loss)
                        self.trade_data[trade] = {'high': self.data.High[-1], 'low': self.data.Low[-1]}
                        print(f"🌙🚀 MOON ENTRY LONG | Price: {current_close:.2f} | Size: {size} | SL: {stop_loss:.2f}")

                elif current_close < self.bb_lower[-1]:
                    risk_amount = self.risk_pct * self.equity
                    stop_loss = current_close + (self.atr_multiplier * current_atr)
                    size = int(round(risk_amount / (self.atr_multiplier * current_atr)))
                    if size > 0:
                        trade = self.sell(size=size, sl=stop_loss)
                        self.trade_data[trade] = {'high': self.data.High[-1], 'low': self.data.Low[-1]}
                        print(f"🌙🌑 MOON ENTRY SHORT | Price: {current_close:.2f} | Size: {size} | SL: {stop_loss:.2f}")

        # Moon Dev Trailing Stop Management 🌙🛡️
        for trade in self.trades:
            if not trade.is_open or trade not in self.trade_data:
                continue
            
            if trade.is_long:
                current_high = self.data.High[-1]
                if current_high > self.trade_data[trade]['high']:
                    self.trade_data[trade]['high'] = current_high
                    print(f"🌙📈 NEW LONG HIGH: {current_high:.2f}")
                
                new_sl = self.trade_data[trade]['high'] -