import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping and formatting
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'DateTime'
}, inplace=True)

data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

class VortexChopTrend(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    stop_pct = 0.02      # 2% stop loss
    
    def init(self):
        # Calculate indicators using TA-Lib and pandas_ta
        self.vi_plus, self.vi_minus = self.I(talib.VORTEX, 
                                            self.data.High, 
                                            self.data.Low, 
                                            self.data.Close, 14)
        
        self.ci = self.I(self._calculate_ci, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 14)
        print("🌙 VortexChopTrend Strategy Initialized! ✨")

    def _calculate_ci(self, high, low, close, length):
        # Convert numpy arrays to pandas Series for pandas_ta
        ci = pandas_ta.chop(high=pd.Series(high), 
                           low=pd.Series(low), 
                           close=pd.Series(close), 
                           length=length)
        return ci.values

    def next(self):
        # Moon Dev progress tracker
        if len(self.data) % 1000 == 0:
            print(f"🌙 Processing bar {len(self.data)} | Price: {self.data.Close[-1]} | VI+={self.vi_plus[-1]:.2f}, VI-={self.vi_minus[-1]:.2f}, CI={self.ci[-1]:.2f}")

        # Entry logic
        if not self.position:
            if (crossover(self.vi_plus, self.vi_minus) and 
                self.ci[-1] < 38.6):
                
                entry_price = self.data.Close[-1]
                stop_loss = entry_price * (1 - self.stop_pct)
                take_profit = entry_price + 2*(entry_price - stop_loss)
                
                # Position sizing calculation
                risk_amount = self.equity * self.risk_percent
                risk_per_share = entry_price - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=stop_loss, 
                            tp=take_profit,
                            tag="VortexBullishEntry")
                    print(f"🚀🌙 LONG ENTRY @ {entry_price:.2f} | Size: {position_size} | CI: {self.ci[-1]:.2f} ✨")

        # Exit logic
        else:
            if (crossover(self.vi_minus, self.vi_plus) or (self.ci[-1] >= 61):
                self.position.close()
                exit_reason = ('VI-' if crossover(self.vi_minus, self.vi_plus) 
                              else 'CI Threshold')
                print(f"🌑 EXIT @ {self.data.Close[-1]:.2f} | Reason: {exit_reason} 🌙")

# Backtest execution
bt = Backtest(data, VortexChopTrend, cash=1_000_000, commission=.002)
stats = bt.run()
print(stats)
print(stats._strategy)