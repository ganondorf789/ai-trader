I'll fix the code by removing all backtesting.lib imports and replacing crossover functions with manual checks. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Liquidation Cascade Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🚀 Data Preparation Rocket Launch
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🧹 Data Cleaning Magic
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'FundingRate'  # Assuming funding rate column exists
}, inplace=True)

class LiquidationDivergence(Strategy):
    # 🌟 Strategy Configuration
    risk_pct = 0.01  # 1% risk per trade
    bb_period = 20
    rsi_period = 14
    atr_period = 14
    
    def init(self):
        # 📈 Technical Indicators Setup
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            lambda close: talib.BBANDS(close, self.bb_period, 2, 2),
            self.data.Close, name=['UpperBB', 'MiddleBB', 'LowerBB']
        )
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.funding_ma = self.I(talib.SMA, self.data.FundingRate, 5)
        self.price_ma = self.I(talib.SMA, self.data.Close, 5)
        
        self.entry_bar = 0  # 📆 Track entry timing
        print("🌌 Moon Dev Indicators Initialized! All systems go! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        
        # 🌙 Moon Dev Exit Conditions
        if self.position:
            # Replaced crossover with manual check
            rsi_cross_up = (self.rsi[-2] <= 35) and (self.rsi[-1] > 35)
            exit_conditions = (
                rsi_cross_up or
                current_close > self.lower_bb[-1] or
                self.data.FundingRate[-1] <= self.funding_ma[-1] or
                (len(self.data) - self.entry_bar) >= 3
            )
            if exit_conditions:
                self.position.close()
                print(f"🌑 Moon Exit: Closed at {current_close:.2f} | RSI {self.rsi[-1]:.1f} 🌗")
                print(f"🌠 Moon Beam Analysis: Position held for {len(self.data) - self.entry_bar} bars ✨")

        # 🚀 Moon Dev Entry Conditions
        else:
            funding_divergence = (self.data.FundingRate[-1] > self.funding_ma[-1]) and (current_close < self.price_ma[-1])
            bb_break = current_close < self.lower_bb[-1]
            # Replaced RSI crossover with manual check
            rsi_trigger = (self.rsi[-1] < 30) and (self.rsi[-2] >= 30)
            volume_spike = self.data.Volume[-1] > self.volume_ma[-1]
            
            if all([funding_divergence, bb_break, rsi_trigger, volume_spike]):
                # 💰 Risk Management Calculation
                bb_width = self.upper_bb[-1] - self.lower_bb[-1]
                stop_loss = current_close + 2 * bb_width
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (stop_loss - current_close)))
                
                if position_size > 0 and self.data.Volume[-1] > self.volume_ma[-1]:
                    self.sell(size=position_size, sl=stop_loss)
                    self.entry_bar = len(self.data)
                    print(f"🌕 Moon Entry: Short