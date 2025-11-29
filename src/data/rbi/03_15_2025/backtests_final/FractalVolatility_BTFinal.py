Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
from backtesting import Backtest, Strategy
import talib
import pandas as pd

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class FractalVolatility(Strategy):
    risk_percentage = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI_14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=14, name='ATR_SMA_14')
        
        # Fractal detection indicators
        self.max_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='FRACTAL_HIGH')
        self.min_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='FRACTAL_LOW')

    def next(self):
        # Get indicator values
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]
        current_atr_sma = self.atr_sma[-1]
        
        # Fractal calculations (2-bar confirmation lag)
        high_two_back = self.data.High[-3]  # High from 2 bars ago
        low_two_back = self.data.Low[-3]    # Low from 2 bars ago
        current_max = self.max_high[-1]     # Current 5-period MAX
        current_min = self.min_low[-1]     # Current 5-period MIN

        # Entry condition flags
        bullish_fractal = (low_two_back == current_min) and not pd.isna(current_min)
        bearish_fractal = (high_two_back == current_max) and not pd.isna(current_max)
        volatility_rising = current_atr > current_atr_sma

        # Moon Dev Risk Management 🌙
        if not self.position:
            # Long entry constellation 🌌
            if current_rsi < 30 and volatility_rising and bullish_fractal:
                entry_price = self.data.Close[-1]
                sl_price = low_two_back
                
                # Calculate position size
                risk_per_share = entry_price - sl_price
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_percentage
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = entry_price + 2 * risk_per_share
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🌙 MOON DEV LONG SIGNAL 🌕")
                        print(f"Entry: {entry_price:.2f} | Size: {position_size} BTC")
                        print(f"RSI: {current_rsi:.2f} | ATR Strength: {current_atr:.2f} ✨")
                        print(f"Stop Loss: {sl_price:.2f} | Take Profit: {tp_price:.2f} 🌌")

            # Short entry constellation 🌑
            elif current_rsi > 70 and volatility_rising and bearish_fractal:
                entry_price = self.data.Close[-1]
                sl_price = high_two_back
                
                # Calculate position size
                risk_per_share = sl_price - entry_price
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_percentage
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = entry_price - 2 * risk_per_share
                        self.sell(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🌙 MOON DEV SHORT SIGNAL 🌑")
                        print(f"Entry: {entry