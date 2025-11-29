Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙✨
def prepare_data(filepath):
    data = pd.read_csv(filepath)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
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
    return data

# Moon Dev Strategy Implementation 🚀🌙
class VwapBollingerReversion(Strategy):
    # Strategy Parameters
    bb_period = 20
    bb_dev = 2
    vwap_slope_window = 5
    ci_length = 14
    atr_period = 14
    min_bb_width = 0.05  # 5% minimum bandwidth
    
    def init(self):
        # Calculate Bollinger Bands 🌈
        close = self.data.Close
        self.bb_upper = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_upper', which=0)
        self.bb_middle = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_middle', which=1)
        self.bb_lower = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_lower', which=2)
        
        # Calculate VWAP with pandas_ta 📊
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP')
        
        # Choppiness Index 🌪️
        self.ci = self.I(ta.chop, self.data.High, self.data.Low, self.data.Close, self.ci_length, name='Choppiness')
        
        # ATR for risk management 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
    def next(self):
        # Moon Dev Entry Logic 🌙✨
        if not self.position:
            # Check all entry conditions
            if (self.data.Close[-1] < self.bb_lower[-1] and
                self.vwap[-1] > self.vwap[-self.vwap_slope_window] and
                (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_middle[-1] >= self.min_bb_width):
                
                ci = self.ci[-1]
                # Dynamic Risk Sizing 📉📈
                if ci > 61.8:
                    risk_pct = 0.5
                elif ci < 38.2:
                    risk_pct = 2.0
                else:
                    risk_pct = 1.0
                
                # Calculate Position Size ⚖️
                risk_amount = self.equity * risk_pct / 100
                atr_value = self.atr[-1]
                stop_distance = 1.5 * atr_value
                entry_price = self.data.Close[-1]
                risk_per_share = stop_distance
                
                if risk_per_share == 0:
                    print("🌙 Zero risk detected! Skipping trade.")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    # Execute Moon Dev Entry 🚀
                    self.buy(size=position_size)
                    self.tp_price = self.bb_upper[-1]
                    self.sl_price = entry_price - stop_distance
                    print(f"🚀🌙 MOON DEV LONG! Size: {position_size}")
                    print(f"   Entry: {entry_price:.2f}, CI: {ci:.2f}, TP: {self.tp_price:.2f}, SL: {self.sl_price:.2f}")
        
        # Moon Dev Exit Logic 🌙🎯
        else: