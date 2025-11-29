# 🌙 Moon Dev's Volatility Breakout Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatilityBreakout(Strategy):
    # Strategy parameters
    ema_50_period = 50
    ema_200_period = 200
    atr_period = 14
    atr_sma_period = 20
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Calculate indicators using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_200_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)
        
    def next(self):
        # Wait for sufficient data 🌙
        if len(self.data) < self.ema_200_period:
            return

        # Current indicator values ✨
        current_atr = self.atr[-1]
        current_atr_sma = self.atr_sma[-1]
        low_volatility = current_atr < current_atr_sma
        
        # EMA crossover detection 🚀
        ema_crossover = crossover(self.ema50, self.ema200)
        
        # Long entry conditions 🌙
        if not self.position and ema_crossover[-1] and low_volatility:
            # Calculate position size with proper rounding
            equity = self.equity
            risk_amount = equity * self.risk_percent
            entry_price = self.data.Close[-1]
            risk_per_share = 2 * current_atr  # 2x ATR stop
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                # Set risk management levels ✨
                sl_price = entry_price - 2 * current_atr
                tp_price = entry_price + 1.5 * current_atr
                
                # Execute trade with Moon Dev flair 🚀
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌙✨🚀 MOON DEV LONG ENTRY 🚀✨🌙 | Price: {entry_price:.2f} | Size: {position_size} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | Equity: {equity:.2f}")

    def on_trade_end(self, trade):
        # Moon-themed exit logging 🌙
        print(f"🌙✨🚀 MOON DEV EXIT 🚀✨🌙 | Reason: {trade.exit_comment} | PnL: {trade.pl:.2f} | Equity: {self.equity:.2f}")

# Data preparation with proper cleaning 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set index 🚀
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Launch backtest with Moon Dev parameters 🌙
bt = Backtest(data, VolatilityBreakout, cash=1_000_000, commission=.002)
stats = bt.run()

# Print full Moon Dev analytics ✨
print("\n🌙🌙🌙 MOON DEV FINAL STATS 🌙🌙🌙")
print(stats)
print("\n🚀🚀🚀 STRATEGY DETAILS 🚀🚀🚀")
print(stats._strategy)