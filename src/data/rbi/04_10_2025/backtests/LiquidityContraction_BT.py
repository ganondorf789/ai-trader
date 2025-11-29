```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidityContraction(Strategy):
    def init(self):
        # Volatility indicators
        self.atr3 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 3, name='ATR3')
        self.atr20 = self.I(talib.SMA, self.atr3, 20, name='ATR20_SMA')
        
        # Liquidation zones (swing high/low proxies)
        self.swing_high = self.I(talib.MAX, self.data.High, 50, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, 50, name='SWING_LOW')
        
        print("🌙 MOON DEV BACKTEST INITIALIZED ✨ | ATR3:", self.atr3[-1], "| Swing High:", self.swing_high[-1])

    def next(self):
        if len(self.data) < 50:  # Wait for indicators to warm up
            return

        close = self.data.Close[-1]
        atr3 = self.atr3[-1]
        atr20 = self.atr20[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # Moon-themed debug prints
        print(f"🌙 {self.data.index[-1]} | Close: {close:.2f} | ATR3: {atr3:.2f} vs ATR20: {atr20:.2f}")

        if not self.position:
            # Long entry logic
            if (close <= swing_low * 1.01) and (atr3 < atr20):
                risk_pct = 0.01  # 1% risk per trade
                sl_price = swing_low - 2 * atr3
                risk_amount = self.equity * risk_pct
                position_size = risk_amount / (close - sl_price)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, 
                            tp=close + 3*atr3,
                            tag="MOON LONG 🌕")
                    print(f"🚀 MOON DEV LONG ENTRY 🌙 | Size: {position_size} | SL: {sl_price:.2f}")

            # Short entry logic
            elif (close >= swing_high * 0.99) and (atr3 < atr20):
                risk_pct = 0.01
                sl_price = swing_high + 2 * atr3
                risk_amount = self.equity * risk_pct
                position_size = risk_amount / (sl_price - close)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl_price,
                             tp=close - 3*atr3,
                             tag="MOON SHORT 🌑")
                    print(f"🌑 MOON DEV SHORT ENTRY 🌙 | Size: {position_size} | SL: {sl_price:.2f}")

        else:
            # Volatility expansion exit
            if crossover(self.atr3, self.atr20):
                self.position.close()
                print(f"🌪️ VOLATILITY EXPANSION EXIT | ATR3: {atr3:.2f} > ATR20: {atr20:.2f}")

            # Liquidity grab exit
            if (self.position.is_long and close < swing_low) or \
               (self.position.is_short and close > swing_high):
                self.position.close()
                print(f"💧 LIQUIDITY GRAB EXIT | Price: {close:.2f}")

# Execute backtest with $1M capital
bt = Backtest(data, LiquidityContraction, cash=1_000