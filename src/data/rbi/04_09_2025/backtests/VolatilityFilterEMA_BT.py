```python
# 🌙 Moon Dev's VolatilityFilterEMA Backtest Implementation 🚀
from backtesting import Backtest, Strategy
import pandas as pd
import talib

# ======================
# DATA PREPARATION 🌍📊
# ======================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'

# Load and clean data
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py format 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ======================
# STRATEGY CLASS 🚀💡
# ======================
class VolatilityFilterEMA(Strategy):
    ema_short = 50
    ema_long = 200
    atr_period = 14
    atr_sma_period = 20
    risk_pct = 0.02  # 2% risk per trade 🌙

    def init(self):
        # ======================
        # INDICATORS 📈🔢
        # ======================
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_short, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_long, name='EMA200')
        
        # ATR with proper column mapping 🌙
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=self.atr_period,
                         name='ATR14')
        
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period, name='ATR_SMA20')

    def next(self):
        # ======================
        # TRADING LOGIC ⚖️💹
        # ======================
        if len(self.ema50) < 2 or len(self.ema200) < 2:
            return  # Wait for indicators to warm up 🌙

        # Current and previous values for crossover detection
        curr_ema50 = self.ema50[-1]
        curr_ema200 = self.ema200[-1]
        prev_ema50 = self.ema50[-2]
        prev_ema200 = self.ema200[-2]

        # Entry Conditions 🌙✨
        if not self.position:
            # EMA Crossover Detection
            golden_cross = (prev_ema50 <= prev_ema200) and (curr_ema50 > curr_ema200)
            
            # Volatility Filter
            low_volatility = self.atr[-1] < self.atr_sma[-1]

            if golden_cross and low_volatility:
                # ======================
                # POSITION SIZING 📏💰
                # ======================
                entry_price = self.data.Close[-1]
                stop_loss = entry_price * (1 - self.risk_pct)
                risk_per_share = entry_price - stop_loss
                position_size = (self.risk_pct * self.equity) / risk_per_share
                position_size = int(round(position_size))  # CRITICAL FIX 🌙

                # Moon Dev Entry Signal 🚀
                print(f"\n🌙✨ GOLDEN CROSS DETECTED! ✨")
                print(f"EMA50 ({curr_ema50:.2f}) > EMA200 ({curr_ema200:.2f})")
                print(f"ATR ({self.atr[-1]:.2f}) < SMA20 ({self.atr_sma[-1]:.2f})")
                print(f"🚀 ENTRY: {entry_price:.2f} | SIZE: {position_size} units")

                self.buy(size=position_size, sl=stop_loss)

        # Exit Conditions 🚨
        else:
            if self.atr[-1] > self.atr_sma[-1]:
                # Moon Dev Exit Signal 🌧️
                print(f"\n🌧️ VOLATILITY SPIKE DETECTED!")
                print(f"ATR ({self.atr[-1]:.2f}) > SMA20 ({self.atr_sma[-1]:.2f