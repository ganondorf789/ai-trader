```python
import pandas as pd
import pandas_ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = [col.strip().lower() for col in data.columns]
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Column mapping
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_map, inplace=True)

# Handle datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Add 4h close flag
data['4h_close_flag'] = 0
resampler = data.resample('4H')
for _, group in resampler:
    if not group.empty:
        data.loc[group.index[-1], '4h_close_flag'] = 1

class FundingCloudBreak(Strategy):
    lookback_funding = 2880  # 30 days in 15m intervals
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Funding rate indicators
        self.funding_rate = self.data['Funding']
        self.funding_pct = self.I(
            lambda x: pd.Series(x).rolling(self.lookback_funding).quantile(0.9).values,
            self.funding_rate,
            name='Funding_Pct_90'
        )
        
        # Ichimoku Cloud
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        ichimoku = pandas_ta.ichimoku(high, low, close, 9, 26, 52)
        self.ichimoku_df = ichimoku[0]
        
        # Shift Senkou spans back by 26 periods
        self.senkou_a = self.I(
            lambda: self.ichimoku_df['ISA_9'].shift(-26).dropna().values,
            name='Senkou_A'
        )
        self.senkou_b = self.I(
            lambda: self.ichimoku_df['ISB_26'].shift(-26).dropna().values,
            name='Senkou_B'
        )
        
        # Kijun-sen (baseline)
        self.kijun = self.I(
            lambda: self.ichimoku_df['IKS_26'].values,
            name='Kijun_Sen'
        )
        
        # ATR
        self.atr = self.I(
            talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name='ATR'
        )
        
    def next(self):
        if self.position:
            # Check primary exit condition
            if self.data['4h_close_flag'][-1]:
                if self.data.Close[-1] > self.kijun[-1]:
                    self.position.close()
                    print(f"🌙✨ Moon Exit: 4H Close Above Kijun ({self.kijun[-1]:.2f})!")
        else:
            # Validate we have enough data
            if len(self.data) < 26 or pd.isna(self.senkou_a[-1]) or pd.isna(self.senkou_b[-1]):
                return
            
            # Entry conditions
            funding_condition = self.funding_rate[-1] > self.funding_pct[-1]
            price_condition = (self.data.Close[-1] < self.senkou_a[-1] and 
                              self.data.Close[-1] < self.senkou_b[-1])
            
            if funding_condition and price_condition:
                atr_val = self.atr[-1]
                if atr_val == 0:
                    return
                
                # Risk management
                risk_amount = self.equity * self.risk_pct
                stop_distance = 2 * atr_val
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    # Short entry with stop and target
                    entry_price = self.data.Close[-1]
                    self.sell(
                        size=position_size,
                        sl=entry_price + stop_distance,
                        tp=entry_price - 2*stop_distance
                    )
                    print(f"🌙