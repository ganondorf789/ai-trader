```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta

# Load and preprocess data
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class ContangoFade(Strategy):
    def init(self):
        # Calculate indicators with proper self.I() wrapping
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Bollinger Bands components
        self.upper_band = self.I(lambda: talib.BBANDS(self.data.Close, 20, 2, 2)[0])
        self.lower_band = self.I(lambda: talib.BBANDS(self.data.Close, 20, 2, 2)[2])
        self.bb_width = self.I(lambda: self.upper_band - self.lower_band)
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, 20)
        
        # Print Moon Dev initialization message
        print("🌙 Moon Dev Backtest Initialized! Ready to fade contango ✨")

    def next(self):
        current_price = self.data.Close[-1]
        
        if not self.position:
            # Get current indicator values
            current_vix_front = self.data.vix_front[-1]
            current_vix_second = self.data.vix_second[-1]
            current_contango = (current_vix_second - current_vix_front) / current_vix_front * 100
            current_adx = self.adx[-1]
            
            # Entry conditions
            if current_contango > 2 and current_adx < 20:
                # Risk management calculations
                risk_percent = 0.05  # Max 5% allocation
                stop_loss_pct = 0.10  # 10% VIX spike stop
                current_vix = self.data.vix[-1]
                
                position_size = (risk_percent * self.equity) / (current_vix * stop_loss_pct)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.sell(size=position_size, meta={
                        'entry_vix': current_vix,
                        'entry_price': current_price
                    })
                    print(f"🚀 MOON SHORT ENTRY 🚀 Contango: {current_contango:.2f}% | ADX: {current_adx:.2f}")
                    print(f"Size: {position_size} | Entry VIX: {current_vix:.2f} | Moon Power: 100% 🌕")
        else:
            # Get current values for exit checks
            current_vix_front = self.data.vix_front[-1]
            current_vix_second = self.data.vix_second[-1]
            current_contango = (current_vix_second - current_vix_front) / current_vix_front * 100
            bb_width = self.bb_width[-1]
            bb_width_sma = self.bb_width_sma[-1]
            current_vix = self.data.vix[-1]
            entry_vix = self.position.meta['entry_vix']
            
            # Exit conditions
            exit_signal = False
            
            # Bollinger Band expansion exit
            if bb_width > bb_width_sma:
                print(f"✨ BB Width Alert! {bb_width:.2f} > {bb_width_sma:.2f} - Preparing lunar escape 🚀")
                exit_signal = True
            
            # Contango flip exit
            if current_contango <= 0:
                print(f"🌌 Contango Flip Detected! Current: {current_contango:.2f}% - Initiating dark matter protocol 🌑")
                exit_signal = True
                
            # VIX spike stop loss
            if current_vix >= entry_vix * 1.10:
                print(f"🌪️ VIX STORM DETECTED! {current_vix:.2f} vs {entry_vix:.2f} - Activating moon shields 🛡️")
                exit_signal = True
                
            if exit_signal:
                self.close()
                pnl