Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and prepare data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricVwapBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Moon Dev Indicator Setup ✨
        def calculate_vwap(h, l, c, v):
            tp = (h + l + c) / 3
            cumulative_tpv = (tp * v).cumsum()
            cumulative_vol = v.cumsum()
            return cumulative_tpv / cumulative_vol
            
        self.vwap = self.I(calculate_vwap, 
                          self.data.High, 
                          self.data.Low, 
                          self.data.Close, 
                          self.data.Volume,
                          name='VWAP')
        
        self.std_dev = self.I(talib.STDDEV, 
                            self.data.Close, 
                            timeperiod=20,
                            name='STDDEV')
        
        self.volume_median = self.I(talib.MEDIAN,
                                  self.data.Volume,
                                  timeperiod=20,
                                  name='VOL_MED')
        
        self.atr = self.I(talib.ATR,
                        self.data.High,
                        self.data.Low,
                        self.data.Close,
                        timeperiod=14,
                        name='ATR')
        
        print("🌙 VolumetricVwapBreakout Strategy Activated! ✨")
        print("🌌 All indicators initialized with pure TA-Lib power!")

    def next(self):
        # Wait for sufficient data 🌙
        if len(self.data) < 21:
            print("🌙 Waiting for enough cosmic data...")
            return
            
        # Previous period values
        prev_close = self.data.Close[-2]
        prev_volume = self.data.Volume[-2]
        prev_vwap = self.vwap[-2]
        prev_std = self.std_dev[-2]
        prev_upper = prev_vwap + 2 * prev_std
        prev_lower = prev_vwap - 2 * prev_std
        prev_vol_med = self.volume_median[-2]
        
        # Current period values
        current_open = self.data.Open[-1]
        current_atr = self.atr[-1]
        
        # Moon Dev Trading Logic 🌙🚀
        if not self.position:
            # Long Entry
            if prev_close > prev_upper and prev_volume > prev_vol_med:
                sl = current_open - current_atr
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_open - sl
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.buy(size=size, 
                            sl=sl,
                            tp=current_open + 2*current_atr)
                    print(f"🌙🚀 LONG SIGNAL DETECTED! Size: {size} | Entry: {current_open:.2f} | SL: {sl:.2f} ✨")
                    print(f"🌕 Price broke above VWAP+2σ with strong volume!")

            # Short Entry
            elif prev_close < prev_lower and prev_volume > prev_vol_med:
                sl = current_open + current_atr
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = sl - current_open
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.sell(size=size,
                             sl=sl,
                             tp=current_open - 2*current_atr)
                    print(f"🌙📉 SHORT SIGNAL DETECTED! Size: {size} | Entry: {current_open:.2f} | SL: {sl:.2f} ✨")
                    print(f"🌑 Price broke below VWAP