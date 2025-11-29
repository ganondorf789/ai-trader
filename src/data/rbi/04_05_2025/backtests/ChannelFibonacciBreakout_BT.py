```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev data preparation magic 🌙✨
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

# Moon Dev's Channel Fibonacci Breakout Strategy 🚀📈
class ChannelFibonacciBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    consecutive_losses = 0
    
    def init(self):
        # 🌙 Bollinger Bands (20, 2)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_LOWER', which=2)
        
        # ✨ Keltner Channel (20, 1.5 ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.ema = self.I(talib.EMA, self.data.Close, 20)
        self.keltner_upper = self.I(lambda: self.ema + self.atr * 1.5, name='KELTNER_UPPER')
        self.keltner_lower = self.I(lambda: self.ema - self.atr * 1.5, name='KELTNER_LOWER')
        
        # 🌊 Hybrid Channel Bands
        self.hybrid_upper = self.I(lambda: (self.bb_upper + self.keltner_upper) / 2, name='HYBRID_UPPER')
        self.hybrid_lower = self.I(lambda: (self.bb_lower + self.keltner_lower) / 2, name='HYBRID_LOWER')
        
        # 📊 Volume confirmation
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOLUME_SMA')
        
        # 🏔️ Swing point detection
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SWING_LOW')

    def next(self):
        # Moon Dev debug prints 🌙🔍
        if len(self.data) % 100 == 0:
            print(f"🌙 Current Close: {self.data.Close[-1]} | Hybrid Upper: {self.hybrid_upper[-1]} | Volume SMA: {self.volume_sma[-1]}")

        if self.consecutive_losses >= 3:
            print("🚫 3 consecutive losses - trading halted!")
            return

        price = self.data.Close[-1]
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        swing_range = self.swing_high[-1] - self.swing_low[-1]

        if not self.position:
            # 🌟 Long entry logic
            if price > self.hybrid_upper[-1] and volume_ok and swing_range > 0:
                sl = self.hybrid_lower[-1]
                risk_distance = price - sl
                if risk_distance <= 0:
                    return
                
                fib_382 = self.swing_high[-1] + swing_range * 0.382
                fib_618 = self.swing_high[-1] + swing_range * 0.618
                position_size = int(round((self.equity * self.risk_pct) / risk_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tag={'fib_382': fib_382, 'fib_618': fib_618})
                    print(f"🚀🌕 LONG ENTRY! Size: {position_size} | SL: {sl} | TP1: {fib_382} | TP2: {fib_618}")

            # 🌟 Short entry logic
            elif price < self.hybrid_lower[-1] and volume_ok and