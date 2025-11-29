I notice the code was cut off at the end. Here's the complete fixed version with all technical issues resolved while preserving the original strategy logic:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class BandwidthImpulse(Strategy):
    def init(self):
        # Bollinger Bands with 20 SMA, 2 std
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0),
            self.data.Close,
            name=['BB_upper', 'BB_middle', 'BB_lower']
        )
        
        # Bollinger Band Width calculation
        self.bb_width = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper,
            self.bb_middle,
            self.bb_lower,
            name='BB_width'
        )
        
        # 6-month low of BB Width (17,280 periods for 15m data)
        self.bb_width_low = self.I(
            talib.MIN,
            self.bb_width,
            timeperiod=17280,
            name='BB_width_low'
        )
        
        # ADX with 14-period
        self.adx = self.I(
            talib.ADX,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=14,
            name='ADX'
        )
        
        self.take_profit = None

    def next(self):
        # Check for sufficient data
        if len(self.adx) < 2 or len(self.bb_width_low) < 1:
            return
        
        current_close = self.data.Close[-1]
        current_bb_middle = self.bb_middle[-1]
        
        # ADX crossover condition
        current_adx = self.adx[-1]
        previous_adx = self.adx[-2]
        adx_cross_above = previous_adx <= 25 and current_adx > 25
        
        # BB Width contraction condition
        current_bb_width = self.bb_width[-1]
        current_bb_width_low = self.bb_width_low[-1]
        bb_contraction = current_bb_width == current_bb_width_low
        
        print(f"🌙 Moon Dev Alert: ADX={current_adx:.1f}, BB%={current_bb_width*100:.2f}%")
        
        # Entry logic
        if not self.position and bb_contraction and adx_cross_above:
            risk_pct = 0.01  # 1% risk per trade
            direction = None
            
            if current_close > current_bb_middle:
                direction = 'LONG'
                sl_price = self.bb_lower[-1]
            elif current_close < current_bb_middle:
                direction = 'SHORT'
                sl_price = self.bb_upper[-1]
            
            if direction:
                risk_amount = self.equity * risk_pct
                risk_per_share = abs(current_close - sl_price)
                
                if risk_per_share <= 0:
                    print("🚨 Moon Dev Warning: Zero risk detected!")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if direction == 'LONG':
                    self.buy(size=position_size)
                    self.take_profit = current_close * (1 + 1.5 * current_bb_width)
                    print(f"🚀 Moon Dev LONG Entry! Size: {position_size}")
                else:
                    self.sell(size=position_size)
                    self.take_profit = current_close * (1 - 1.5 * current_bb_width)
                    print(f"🌪️ Moon Dev SHORT Entry! Size: {position_size}")

        # Exit logic
        if self.position:
            if self.position.is_long:
                if self.data.High[-1] >= self.take_profit:
                    self.position.close()
                    print(f"🎯 Moon Dev Profit Reached! +{self.take_profit/self.position.entry_price-1:.2%}")
                elif self.data.Low[-1] <= self.bb_lower[-1]:
                    self.position.close()
                    print(f"💥 Moon Dev Stop Loss! -{1 - self.bb_lower[-1]/self.position.entry_price:.2%}")
            else:
                if self.data.Low[-1] <= self.take_profit:
                    self.position.close()
                    print(f"🎯 Moon Dev Profit Reached! +{1 - self.take_profit/self.position.entry_price:.