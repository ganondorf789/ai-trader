I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev improvements:

```python
# 🌙 Moon Dev's BandSqueezeReversal Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# 🚀 Data Preparation with Moon Dev Standards
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🌙 Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandSqueezeReversal(Strategy):
    def init(self):
        # ✨ Heikin-Ashi Calculation
        ha_df = self.data.df.ta.heikin_ashi(open='Open', high='High', low='Low', close='Close')
        self.ha_close = self.I(ha_df['HA_close'].values, name='HA_Close')
        self.ha_open = self.I(ha_df['HA_open'].values, name='HA_Open')
        
        # 📊 Bollinger Bands with Width Calculation
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        bb_width = (self.bb_upper - self.bb_lower)/self.bb_middle * 100
        self.bb_width = self.I(bb_width, name='BB_Width')
        
        # 📈 Momentum Indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=5, name='RSI_5')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing_Low')
        
        self.entry_price = None
        self.stop_loss = None

    def next(self):
        # 🌙 Moon Dev Debug Prints
        if len(self.data) % 500 == 0:
            print(f"🌙 MOON DEV CHECK | Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | BB Width: {self.bb_width[-1]:.2f}% | RSI: {self.rsi[-1]:.2f}")
        
        if not self.position:
            # 🚀 Entry Logic
            bb_squeeze = self.bb_width[-1] < 1.5
            ha_bullish = all(self.ha_close[-3:] > self.ha_open[-3:])
            
            if bb_squeeze and ha_bullish:
                # 🔒 Risk Management
                sl_swing = self.swing_low[-1]
                sl_atr = self.data.Close[-1] - self.atr[-1]
                sl = max(sl_swing, sl_atr)
                
                risk_percent = 0.01  # 1% risk
                risk_amount = self.equity * risk_percent
                entry_price = self.data.Open[-1]
                risk_per_share = entry_price - sl
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl)
                        self.entry_price = entry_price
                        self.stop_loss = sl
                        print(f"🚀 MOON DEV LONG ENTRY | Size: {position_size} | Entry: {entry_price:.2f} | SL: {sl:.2f}")
        else:
            # 🎯 Exit Logic
            if self.rsi[-2] < 70 and self.rsi[-1] >= 70:  # RSI crossover above 70
                self.position.close()
                print(f"🎯 MOON DEV RSI EXIT | Price: {self.data.Close[-1]:.2f} | RSI: