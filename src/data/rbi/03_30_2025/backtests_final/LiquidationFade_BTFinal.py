Here's the fully debugged and Moon Dev themed backtest code for the Liquidation Fade strategy:

```python
# 🌙 Moon Dev's Liquidation Fade Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# ========================
# 🛠 DATA PREPARATION
# ========================
print("🌙 INITIALIZING MOON DEV DATA SYSTEMS...")
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data with lunar precision
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    print("🌙 DATA LOAD SUCCESS: Cosmic alignment achieved! ✨")
except Exception as e:
    print(f"🌙 DATA LOAD ERROR: Interstellar interference detected! {e}")

# Column mapping with Moon Dev precision 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})[['Open', 'High', 'Low', 'Close', 'Volume']]

# Simulate order book imbalance (placeholder for real data)
data['imbalance'] = 0.72  # Mock bullish imbalance
print("🌙 ORDER BOOK SYSTEMS ONLINE: Imbalance metrics initialized")

# ========================
# 🚀 LIQUIDATION FADE STRATEGY
# ========================
class LiquidationFade(Strategy):
    vwap_period = 15
    risk_percent = 0.01
    max_daily_trades = 5
    trades_today = 0
    
    def init(self):
        # 🌙 Core Indicators
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low, 
                          close=self.data.Close,
                          volume=self.data.Volume,
                          length=self.vwap_period)
        
        self.price_spike = self.I(talib.MAX,
                                self.data.High,
                                timeperiod=5)  # 5-bar spike window

        print("🌙 MOON DEV INIT COMPLETE: Indicators ready for launch! 🚀")

    def next(self):
        # 🌙 New Day Reset
        if self.data.index[-1].day != self.data.index[-2].day:
            self.trades_today = 0
            print("🌙 NEW DAY RESET: Trade counter refreshed ✨")

        if self.trades_today >= self.max_daily_trades:
            print("🌙 TRADE LIMIT REACHED: Maximum daily trades executed 🌑")
            return

        # 🌙 Current Market Conditions
        current_close = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        price_deviation = abs(current_close - current_vwap) / current_vwap
        
        # 🌙 Liquidation Spike Detection
        if price_deviation > 0.02 and self.data.imbalance[-1] > 0.7:
            if current_close > current_vwap:  # Bullish liquidation spike
                print("🌙 BULLISH SPIKE DETECTED: Preparing short entry 🌘")
                self.enter_short(current_close, current_vwap)
            else:  # Bearish liquidation spike
                print("🌙 BEARISH SPIKE DETECTED: Preparing long entry 🌖")
                self.enter_long(current_close, current_vwap)

        # 🌙 Time-based Exit
        for trade in self.trades:
            if len(self.data) - trade.entry_bar > 2:  # 30min = 2x15m bars
                trade.close()
                print(f"🌙 MOON ALERT: Time Exit at {current_close:.2f} ✨")

    def enter_short(self, price, vwap):
        # 🌙 Risk Calculation
        risk_amount = self.equity * self.risk_percent
        stop_loss = price * 1.03  # 3% stop loss
        position_size = int(round(risk_amount / (stop_loss - price)))
        
        if position_size > 0:
            self.sell(size=position_size, 
                     sl=stop_loss,
                     tp=vwap)
            self.trades_today += 1
            print(f"🌙 SHORT ENTRY 🌑 Price: {price:.2f} | Size: {position_size} 🚀