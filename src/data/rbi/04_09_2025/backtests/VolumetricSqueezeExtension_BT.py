```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricSqueezeExtension(Strategy):
    initial_capital = 1_000_000
    risk_per_trade = 0.02
    max_exposure = 0.10
    daily_loss_limit = 0.05
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # Bollinger Bands
        self.upper_band, self.middle_band, self.lower_band = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name=['Upper BB', 'Middle BB', 'Lower BB'])
        
        # Bollinger Bandwidth calculations
        self.bandwidth = (self.upper_band - self.lower_band) / self.middle_band
        self.bandwidth_sma20 = self.I(talib.SMA, self.bandwidth, 20, name='Bandwidth SMA20')
        
        # Volume indicators
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, 50, name='Volume SMA50')
        current_volume = self.data.Volume
        self.volume_ratio = current_volume / self.volume_sma50
        
        # VW-MACD calculations
        vw_close = self.data.Close * (current_volume / self.volume_sma50)
        self.macd_line, self.macd_signal, _ = self.I(talib.MACD, vw_close, fastperiod=12, slowperiod=26, signalperiod=9, name='VW-MACD')
        
        # Swing points
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
    def next(self):
        # Moon Dev risk management checks 🌙
        if self.equity < self.initial_capital * (1 - self.daily_loss_limit):
            print(f"🌑 MOON DEV ALERT: Daily loss limit hit! Stopping trading.")
            self.position.close()
            return
            
        if len(self.data) < 50:
            return
            
        # Check bandwidth squeeze condition
        bandwidth_threshold = 0.2 * self.bandwidth_sma20[-1]
        squeeze_on = self.bandwidth[-1] < bandwidth_threshold
        
        # Volume confirmation
        volume_ok = self.data.Volume[-1] > 1.2 * self.volume_sma50[-1]
        
        # MACD crossover conditions
        macd_bullish = crossover(self.macd_line, self.macd_signal)
        macd_bearish = crossunder(self.macd_line, self.macd_signal)
        
        # Price position relative to Bollinger
        price_above_middle = self.data.Close[-1] > self.middle_band[-1]
        price_below_middle = self.data.Close[-1] < self.middle_band[-1]
        
        # Calculate current exposure
        total_exposure = sum(pos.size * pos.entry_price for pos in self.positions)
        if total_exposure > self.equity * self.max_exposure:
            return
        
        # Long entry logic 🌕
        if not self.position.is_long and squeeze_on and volume_ok and macd_bullish and price_above_middle:
            # Risk calculation
            entry_price = self.data.Close[-1]
            stop_loss = self.lower_band[-1]
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return
                
            position_size = (self.equity