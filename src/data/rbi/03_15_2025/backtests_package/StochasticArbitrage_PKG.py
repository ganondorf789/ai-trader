Here's the fixed code with all `backtesting.lib` imports and functions removed, and replaced with proper Moon Dev themed implementations:

```python
# 🌙 Moon Dev's StochasticArbitrage Backtest Implementation
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data Preparation 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data according to Moon Dev specifications ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class StochasticArbitrage(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Calculate all indicators using TA-Lib wrapped in self.I() ✨
        self._init_indicators()
        print("🌙 Moon Dev Strategy Activated! Ready for cosmic gains! 🚀")

    def _init_indicators(self):
        # Stochastic Oscillator (14,3,3)
        stoch_k, stoch_d = talib.STOCH(self.data.High, self.data.Low, self.data.Close,
                                      fastk_period=14, slowk_period=3, slowd_period=3)
        self.stoch_k = self.I(lambda: stoch_k, name='Stoch %K')
        self.stoch_d = self.I(lambda: stoch_d, name='Stoch %D')

        # Bollinger Bands (20,2)
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: upper, name='BB Upper')
        self.bb_middle = self.I(lambda: middle, name='BB Middle')
        self.bb_lower = self.I(lambda: lower, name='BB Lower')

        # ATR (14) for risk management 🌙
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')

        # Z-Score Calculation (using price vs 20-period mean)
        sma20 = talib.SMA(self.data.Close, 20)
        spread = self.data.Close - sma20
        z_score = (spread - talib.SMA(spread, 20)) / talib.STDDEV(spread, 20)
        self.z_score = self.I(lambda: z_score, name='Z-Score')

    def next(self):
        # Moon Dev's Cosmic Entry Logic ✨
        if self.position:
            self._manage_exits()
            return

        current_close = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Long Entry Constellation 🌌
        if (self.stoch_k[-1] < 20 and 
            self.z_score[-1] < -2 and 
            current_close <= self.bb_lower[-1]):
            
            position_size = self._calculate_size(current_close, atr)
            if position_size > 0:
                self.buy(size=position_size, sl=current_close - 2*atr)
                print(f"🌙✨ BUY SIGNAL! Size: {position_size} @ {current_close} 🚀")

        # Short Entry Constellation 🌠
        elif (self.stoch_k[-1] > 80 and 
              self.z_score[-1] > 2 and 
              current_close >= self.bb_upper[-1]):
            
            position_size = self._calculate_size(current_close, atr)
            if position_size > 0:
                self.sell(size=position_size, sl=current_close + 2*atr)
                print(f"🌙✨ SELL SIGNAL! Size: {position_size} @ {current_close} 💫")

    def _manage_exits(self):
        # Moon Dev's Exit Protocol 🛑
        current_close = self.data.Close[-1]
        
        # Replace crossover/crossunder with manual checks
        stoch_k_cross_above_50 = self.stoch_k[-2] < 50 and self.stoch_k[-1] > 50
        stoch_k_cross_below_50 = self.stoch_k[-2] > 50 and self