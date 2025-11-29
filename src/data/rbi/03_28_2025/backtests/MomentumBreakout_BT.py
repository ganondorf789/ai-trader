```python
# 🌙 Moon Dev Backtest Implementation for MomentumBreakout Strategy
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# 🚀 Data Preparation Moon-Style
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class MomentumBreakout(Strategy):
    # 🌙 Strategy Parameters
    ema50_period = 50
    ema200_period = 200
    adx_period = 14
    atr_period = 14
    rsi_period = 5
    recent_high_window = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌌 Cosmic Indicator Calculations
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=self.recent_high_window)
        
        self.trailing_high = None  # 🌕 For tracking moon shots

    def next(self):
        # 🌑 Check for active positions
        if self.position.is_long:
            current_high = self.data.High[-1]
            self.trailing_high = max(self.trailing_high, current_high) if self.trailing_high else current_high
            
            # 🛑 Exit Conditions
            stop_loss_price = self.trailing_high - 2 * self.atr[-1]
            current_rsi = self.rsi[-1]
            
            # Trailing Stop Check
            if self.data.Close[-1] < stop_loss_price:
                self.position.close()
                print(f"🌑 Moon Dev Exit! Trail Stop at {stop_loss_price:.2f} | Equity: {self.equity:,.0f} 🌌")
                self.trailing_high = None
                return
                
            # RSI Exit Check
            if current_rsi > 70 or current_rsi < 30:
                self.position.close()
                print(f"🌗 Moon Dev RSI Exit at {current_rsi:.2f} | P/L: {self.position.pl_pct:.2f}% 🚀")
                self.trailing_high = None
                return
        else:
            # 🌟 Entry Conditions
            if len(self.data.Close) < 2:
                return  # Not enough data
            
            # Golden Cross Check
            golden_cross = (self.ema50[-2] <= self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            adx_strong = self.adx[-1] > 25
            price_breakout = self.data.Close[-1] > self.recent_high[-1]
            
            if golden_cross and adx_strong and price_breakout:
                # 💰 Risk-Managed Position Sizing
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr[-1]
                
                if atr_value == 0:
                    return  # Avoid black hole division
                
                position_size = risk_amount / (2 * atr_value)
                position_size = int(round(position_size))
                max_possible = int(self.equity // self.data.Close[-1])
                position_size = min(position_size, max_possible)
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.trailing_high = self.data.High[-1]
                    print(f"🚀 Moon Launch! LONG