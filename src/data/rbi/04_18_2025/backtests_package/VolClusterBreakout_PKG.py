Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data with Moon Dev magic 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names with lunar precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolClusterBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_period = 14
    volume_ma_period = 20
    
    def init(self):
        # Calculate indicators using TA-Lib with cosmic precision 🌌
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        
        # Access external data series
        self.liquidation_level = self.data['liquidation_level']
        self.funding_rate = self.data['funding_rate']
        
        # Track daily trades
        self.trades_today = 0
        self.current_day = None

    def next(self):
        current_time = self.data.index[-1]
        current_day = current_time.date()
        
        # Reset daily trade counter with lunar cycle 🌑🌕
        if current_day != self.current_day:
            self.current_day = current_day
            self.trades_today = 0
            print(f"🌙 NEW MOON CYCLE | Date: {current_day}")

        # Check market hours and trade limits 🌙
        if not (12 <= current_time.hour < 16) or self.trades_today >= 3:
            if self.trades_today >= 3:
                print("🌙🚫 TRADE LIMIT REACHED | Maximum 3 trades per day")
            return

        # Get current values with celestial precision
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        liquidation = self.liquidation_level[-1]
        funding = self.funding_rate[-1]
        atr_value = self.atr[-1]
        volume_ma = self.volume_ma[-1]

        # Long entry conditions 🚀
        if (not self.position and
            price > liquidation and
            volume > 2 * volume_ma and
            funding < -0.0001):
            
            # Calculate position size with lunar gravity 🌕
            risk_amount = self.equity * self.risk_pct
            stop_loss = price - 0.75 * atr_value
            risk_per_share = price - stop_loss
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                # Execute trade with initial stop
                self.buy(size=position_size, sl=stop_loss)
                self.trades_today += 1
                print(f"🚀🌙 MOONSHOT ENTRY | Price: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | ATR: {atr_value:.2f} | Volume: {volume:.0f} vs MA: {volume_ma:.0f}")

        # Manage open positions with cosmic wisdom ✨
        for trade in self.trades:
            if not trade.is_long:
                continue

            # Funding rate reversal exit 💸
            if self.funding_rate[-1] >= 0.0001:
                self.close(trade)
                print(f"🌕💥 FUNDING REVERSAL EXIT | Price: {self.data.Close[-1]:.2f} | PnL: {trade.pl:.2f}")
                continue

            # Dynamic exit management 🛡️
            entry = trade.entry_price
            current_high = self.data.High[-1]
            
            # Trailing stop activation
            if current_high >= entry + atr_value:
                new_sl = current_high - 0.5