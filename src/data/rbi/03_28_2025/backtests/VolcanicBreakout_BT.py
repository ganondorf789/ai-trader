Here's the implementation of the VolcanicBreakout strategy:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import pandas_ta as ta

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Resample to 4H and compute Heikin-Ashi
df_4h = data.resample('4H', closed='right', label='right').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

ha_open, ha_high, ha_low, ha_close = [], [], [], []
for i in range(len(df_4h)):
    if i == 0:
        ha_o = df_4h['Open'].iloc[i]
    else:
        ha_o = (ha_open[i-1] + ha_close[i-1])/2
    ha_c = (df_4h['Open'].iloc[i] + df_4h['High'].iloc[i] + df_4h['Low'].iloc[i] + df_4h['Close'].iloc[i])/4
    ha_h = max(df_4h['High'].iloc[i], ha_o, ha_c)
    ha_l = min(df_4h['Low'].iloc[i], ha_o, ha_c)
    
    ha_open.append(ha_o)
    ha_high.append(ha_h)
    ha_low.append(ha_l)
    ha_close.append(ha_c)

df_4h = df_4h.assign(
    HA_Open=ha_open,
    HA_High=ha_high,
    HA_Low=ha_low,
    HA_Close=ha_close
)

# Calculate indicators
tp = (df_4h['HA_High'] + df_4h['HA_Low'] + df_4h['HA_Close'])/3
df_4h['KC_upper'] = talib.EMA(tp, 20) + 2 * talib.ATR(df_4h['HA_High'], df_4h['HA_Low'], df_4h['HA_Close'], 20)
bb_upper, bb_middle, bb_lower = talib.BBANDS(df_4h['HA_Close'], 20, 2, 2)
df_4h['BB_width'] = (bb_upper - bb_lower)/bb_middle
df_4h['BB_width_SMA20'] = talib.SMA(df_4h['BB_width'], 20)
df_4h['vol_3x'] = 3 * talib.SMA(df_4h['Volume'], 20)

# Merge indicators back to 15m data
data = data.join(df_4h[['HA_Close', 'KC_upper', 'BB_width', 'BB_width_SMA20', 'vol_3x']], how='left').ffill()
data['ATR14'] = talib.ATR(data['High'], data['Low'], data['Close'], 14)
data.dropna(inplace=True)

class VolcanicBreakout(Strategy):
    ha_lows = []
    last_ha_low = None
    
    def init(self):
        pass
    
    def next(self):
        current_ha_low = self.data.HA_Close[-1]  # Simplified for demo
        
        # Track HA lows
        if current_ha_low != self.last_ha_low:
            self.ha_lows.append(current_ha_low)
            self.ha_lows = self.ha_lows[-3:]
            self.last_ha_low = current_ha_low
        
        # Entry logic
        if not self.position:
            cond1 = self.data.HA_Close[-1] > self.data.KC_upper[-1]
            cond2 = self.data.BB_width[-1] < self.data.BB_width_SMA20[-1]
            cond3 = self.data.Volume[-1] >= self.data.vol_3x[-1]
            
            if cond1 and cond2 and cond3:
                # Risk management