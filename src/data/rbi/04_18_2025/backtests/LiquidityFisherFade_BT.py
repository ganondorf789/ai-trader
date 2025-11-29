```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION: LiquidityFisherFade Strategy 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# 🌌 DATA PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidityFisherFade(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌊
    fisher_length = 9       # Fisher Transform period 🌗
    bb_length = 20          # Bollinger Bands period 📈
    swing_period = 20       # Swing high/low period 🌓
    cluster_tolerance = 0.001  # 0.1% price tolerance 🌐
    
    def init(self):
        # 🌗 MEDIAN PRICE CHANNEL: (H+L+C)/3
        self.median_price = (self.data.High + self.data.Low + self.data.Close) / 3
        
        # 📉 BOLLINGER BANDS ON MEDIAN PRICE
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, self.median_price, 
            timeperiod=self.bb_length, nbdevup=2, nbdevdn=2,
            matype=talib.MA_Type.SMA
        )
        
        # 🎣 FISHER TRANSFORM WITH PANDAS_TA
        def compute_fisher(high, low, length):
            fisher_df = ta.fisher(high=high, low=low, length=length)
            return fisher_df.iloc[:, 0], fisher_df.iloc[:, 1]
            
        self.fisher, self.fisher_signal = self.I(
            compute_fisher, self.data.High, self.data.Low, 
            self.fisher_length, name=['FISHER', 'FISHER_SIGNAL']
        )
        
        # 🏔️ SWING HIGHS/LOWS FOR LIQUIDITY CLUSTERS
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
    def next(self):
        current_close = self.data.Close[-1]
        
        # 🌗 EXISTING POSITION MANAGEMENT
        if self.position:
            if self.position.is_long:
                # 🌕 LONG EXIT: Close above upper BB or Fisher crosses 0
                if current_close > self.upper_band[-1]:
                    print(f"🚀🌕 MOON EXIT LONG | Price {current_close:.2f} > Upper BB {self.upper_band[-1]:.2f}")
                    self.position.close()
                elif crossover(self.fisher, 0):
                    print(f"🌑🐻 BEARISH FISHER CROSS | Closing LONG")
                    self.position.close()
                    
            elif self.position.is_short:
                # 🌑 SHORT EXIT: Close below lower BB or Fisher crosses 0
                if current_close < self.lower_band[-1]:
                    print(f"🌌🌑 MOON EXIT SHORT | Price {current_close:.2f} < Lower BB {self.lower_band[-1]:.2f}")
                    self.position.close()
                elif crossunder(self.fisher, 0):
                    print(f"🌕🐂 BULLISH FISHER CROSS | Closing SHORT")
                    self.position.close()
        else:
            # 🌓 CALCULATE TRADE PARAMETERS
            fisher_value = self.fisher[-1]
            prev_fisher = self.fisher[-2]
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            
            # 🌑 SHORT ENTRY: Fisher > 2.5 & High near swing high
            if fisher_value > 2.5 and prev_fisher <=