Here's the debugged version of your Volatility Squeeze Breakout strategy with Moon Dev improvements 🌙✨:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas_ta as ta

# Moon Dev Data Preparation 🌙✨
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
    # Proper column mapping
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns={k:v for k,v in column_map.items() if k in data.columns}, inplace=True)
    # Set datetime index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

class VolatilitySqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    
    def init(self):
        # Calculate indicators using TA-Lib 🌟
        close = self.data.Close
        volume = self.data.Volume
        
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0),
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Bollinger Band Width
        bb_width = self.bb_upper - self.bb_lower
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')
        
        # BB Width Percentile (20th percentile over 100 periods)
        self.bb_width_percentile = self.I(
            lambda: ta.quantile(bb_width, length=100, q=0.2),
            name='BB_Width_Percentile'
        )
        
        # Volume MA (20 periods)
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=20, name='Volume_MA')
        
        # Initialize consolidation tracking variables 🌌
        self.in_consolidation = False
        self.consolidation_start = None
        self.cumulative_pv = 0
        self.cumulative_vol = 0
        self.closes_in_consolidation = []

    def _enter_trade(self, direction):
        """Moon Dev Trade Entry Protocol 🌙"""
        price = self.data.Close[-1]
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        
        if direction == 'long':
            stop_price = min(self.closes_in_consolidation[-5:])  # Last 5 closes in consolidation
            take_profit = self._calculate_vwap() + np.std(self.closes_in_consolidation)
            size = risk_amount / (price - stop_price)
            print(f"🌕 MOON DEV LONG SIGNAL 🌕 | Entry: {price:.2f} | Stop: {stop_price:.2f} | Target: {take_profit:.2f}")
        else:
            stop_price = max(self.closes_in_consolidation[-5:])  # Last 5 closes in consolidation
            take_profit = self._calculate_vwap() - np.std(self.closes_in_consolidation)
            size = risk_amount / (stop_price - price)
            print(f"🌑 MOON DEV SHORT SIGNAL 🌑 | Entry: {price:.2f} | Stop: {stop_price:.2f} | Target: {take_profit:.2f}")
        
        # Ensure proper position sizing 🌙
        size = min(size, 0.99)  # Never risk more than 99% of equity
        self.buy(size=size, sl=stop_price, tp=take_profit) if direction == 'long' else self.sell(size=size, sl=stop_price, tp=take_profit)

    def _calculate_vwap(self):
        """Moon Dev VWAP Calculation 🌙✨"""
        if self.cumulative_vol > 0:
            return self.cumulative_pv / self.cumulative_vol
        return self.data.Close[-1]  # Fallback to current price if no volume

    def next(self):