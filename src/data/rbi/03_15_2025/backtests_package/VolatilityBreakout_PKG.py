I've checked the code and found that it uses `backtesting.lib.crossover`. I'll fix this and ensure all indicators are properly implemented without any `backtesting.lib` imports. Here's the corrected version:

```python
# 🌙 Moon Dev's Volatility Breakout Strategy Backtest 🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatilityBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌟 Bollinger Bands Indicator
        def upper_bb(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        self.upper_band = self.I(upper_bb, self.data.Close, name='Upper BB')
        
        # 🌠 MACD Histogram Calculation
        def macd_hist(close):
            _, _, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return hist
        self.macd_hist = self.I(macd_hist, self.data.Close, name='MACD Hist')
        
        # 📈 Volatility Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR SMA')
        
        # 🛡️ Swing Low for Stop Loss
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        print("🌙 VolatilityBreakout Strategy Initialized with Moon Dev Power! 🚀")

    def next(self):
        # 🌌 Check for entry signals
        if not self.position:
            # Entry Condition 1: Price crosses above Upper BB
            price_break = self.data.Close[-2] <= self.upper_band[-2] and self.data.Close[-1] > self.upper_band[-1]
            
            # Entry Condition 2: MACD Hist positive
            macd_bullish = self.macd_hist[-1] > 0
            
            # Entry Condition 3: Volatility filter
            volatility_ok = self.atr[-1] > self.atr_sma[-1]
            
            if price_break and macd_bullish and volatility_ok:
                # 🧮 Risk Management Calculations
                entry_price = self.data.Open[-1]  # Next bar's open price
                stop_loss = self.swing_low[-1]
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    # 🚀 Execute Buy Order with Stop Loss
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌕 MOON ENTRY! Size: {position_size} @ {entry_price:.2f} | SL: {stop_loss:.2f} 🚀")
        
        else:
            # 🌑 Check exit conditions
            exit_price = self.data.Close[-2] >= self.upper_band[-2] and self.data.Close[-1] < self.upper_band[-1]
            exit_macd = self.macd_hist[-2] >= 0 and self.macd_hist[-1] < 0
            exit_vol = self.atr[-1] < self.atr_sma[-1]
            
            if exit_price or exit_macd or exit_vol:
                self.position.close()
                print(f"🌒 MOON EXIT! P/L: {self.position.pl:.2f} | Equity: {self.equity:.2f} ✨")

# 🛠️ Data Preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# 🧹 Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_dat