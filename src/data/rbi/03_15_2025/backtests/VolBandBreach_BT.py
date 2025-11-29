Here's the complete implementation of the VolBandBreach strategy following your requirements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation and cleaning
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolBandBreach(Strategy):
    def init(self):
        # Bollinger Bands indicator calculation 🌙
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_UPPER')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_LOWER')
        
        # Volume analysis indicators 📊
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=96, name='VOL_SMA')
        
        # Volatility analysis indicators 🌪️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        self.entry_bar = 0  # Track entry bar for holding period

    def next(self):
        # Moon Dev progress tracker 🌙
        if len(self.data) % 500 == 0:
            print(f"🌙 Processing bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | Equity: {self.equity:,.2f}")
        
        if not self.position:
            # Check minimum data requirements
            if len(self.volume_sma) < 2 or len(self.bb_upper) < 2:
                return
        
            # Entry conditions check 🔍
            prev_close = self.data.Close[-2]
            prev_upper = self.bb_upper[-2]
            prev_volume = self.data.Volume[-2]
            avg_vol = self.volume_sma[-2]
            current_atr = self.atr[-1]
            
            if (prev_close > prev_upper and
                prev_volume >= 1.5 * avg_vol and
                current_atr < 0.03 * self.data.Close[-1]):
                
                # Calculate risk parameters 🛡️
                stop_price = self.bb_lower[-2]
                entry_price = self.data.Open[-1]
                risk_per_unit = entry_price - stop_price
                
                if risk_per_unit <= 0:
                    print("⚠️🌙 Invalid risk calculation - skipping trade")
                    return
                
                # Position sizing calculation 🧮
                risk_amount = 0.01 * self.equity
                position_size = int(round(risk_amount / risk_per_unit))
                
                if position_size > 0:
                    # Execute trade with stop loss 🚀
                    self.buy(size=position_size, sl=stop_price)
                    self.entry_bar = len(self.data) -1
                    print(f"\n🚀🌙 MOON DEV TRADE LAUNCH 🚀"
                          f"\nEntry: {entry_price:.2f} | Size: {position_size}"
                          f"\nStop Loss: {stop_price:.2f} | Risk/Reward: {risk_per_unit:.2f}"
                          f"\nEquity: {self.equity:,.2f}\n")
        else:
            # Exit conditions check 🛑
            days_in_trade = (len(self.data) -1 - self.entry_bar) / 96
            if days_in_trade >= 2:
                self.position.close()
                print(f"\n🕒🌙 MOON DEV TIME EXIT 🕒"
                      f"\nHeld for {days_in_trade:.1f} days"
                      f"\nExit Price: {self.data.Close[-1]:.