```python
# 🌙 Moon Dev's VoltaicSqueeze Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🚀 DATA PREPARATION 🚀
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping with Moon Dev precision 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VoltaicSqueeze(Strategy):
    risk_percent = 0.02  # 🌙 2% risk per trade
    
    def init(self):
        # 🌗 INDICATOR CALCULATION 🌗
        # Bollinger Bands with 20,2 configuration
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        )
        
        # Volatility squeeze detection 🌌
        self.bb_width = self.I(
            lambda upper, lower: upper - lower, 
            self.bb_upper, self.bb_lower, 
            name='BB_WIDTH'
        )
        self.bb_width_min = self.I(
            talib.MIN, self.bb_width, 
            timeperiod=30, name='BB_WIDTH_MIN'
        )
        
        # Dynamic exit mechanism 🛑
        self.atr = self.I(
            talib.ATR, self.data.High, 
            self.data.Low, self.data.Close, 
            timeperiod=14, name='ATR'
        )
        
        self.highest_high = None  # For trailing stop calculation

    def next(self):
        # 🌙 MOON DEV SIGNAL PROCESSING 🌙
        if not self.position:
            # Entry Logic: Volatility Squeeze Breakout 🚀
            squeeze_condition = (self.bb_width[-1] == self.bb_width_min[-1])
            breakout_condition = (self.data.Close[-1] > self.bb_upper[-1])
            
            if squeeze_condition and breakout_condition:
                # 🌗 RISK MANAGEMENT CALCULATIONS 🌗
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                
                if atr_value == 0:
                    return  # Avoid division by zero
                
                # Position sizing with Moon Dev precision 🌙
                risk_per_share = 2 * atr_value
                estimated_price = self.data.Close[-1]
                max_equity_size = self.equity // estimated_price
                position_size = risk_amount / risk_per_share
                position_size = min(position_size, max_equity_size)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🚀🌙 MOON ALERT: Long Entry @ {self.data.Open[-1]} "
                          f"| Size: {position_size} units | "
                          f"Equity: {self.equity:,.2f} ✨")

        else:
            # 🌗 DYNAMIC EXIT LOGIC 🌗
            current_high = self.data.High[-1]
            self.highest_high = max(self.highest_high, current_high) if self.highest_high else current_high
            
            # Trailing stop calculation
            trailing_stop = self.highest_high - 2 * self.atr[-1]
            
            # Exit Logic: Trailing Stop Trigger 🛑
            if self.data.Low[-1] <= trailing_stop:
                self.sell()
                print(f"🌙✨ MOON EXIT: Closing Position @ {self.data.Close[-1]} "
                      f"| Trailing Stop: {trailing_stop:.2f} "
                      f"| Profit: {self.position.pl_pct:.2%} 🛑")

# 🌙 BACKTEST EX