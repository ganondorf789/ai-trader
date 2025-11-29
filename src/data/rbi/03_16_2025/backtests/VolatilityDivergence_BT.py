```python
# 🌙 Moon Dev's Volatility Divergence Backtest 🌙
import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

# Clean and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Data Cleaning Ritual
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
data = data.set_index('datetime')

class VolatilityDivergence(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌕
    
    def init(self):
        # 🌀 MACD Divergence Indicators
        def _macd(close):
            macd, signal, _ = talib.MACD(close, 
                fastperiod=12, slowperiod=26, signalperiod=9)
            return macd
        self.macd = self.I(_macd, self.data.Close)
        
        # 🌊 Volatility Indicators (ATR instead of VIX)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                        self.data.Close, timeperiod=14)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20)
        
        # 🏔️ Swing High Detection
        self.price_highs = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.macd_highs = self.I(talib.MAX, self.macd, timeperiod=20)
        
        print("🌙 Lunar Indicators Activated! Ready for bearish divergence hunting 🚀")

    def next(self):
        # Avoid early bars without indicator data
        if len(self.data) < 50:
            return
        
        # 🕵️♂️ Divergence Detection Logic
        current_price_high = self.price_highs[-1]
        prev_price_high = self.price_highs[-2]
        current_macd_high = self.macd_highs[-1]
        prev_macd_high = self.macd_highs[-2]
        
        bearish_divergence = (current_price_high > prev_price_high) and \
                            (current_macd_high < prev_macd_high)
        
        # 🌪️ Volatility Spike Check
        volatility_spike = self.atr[-1] > self.atr_sma[-1] * 1.10
        
        # 🚀 ENTRY: Short on bearish divergance + volatility spike
        if not self.position and bearish_divergence and volatility_spike:
            # 🛡️ Risk Management Calculations
            entry_price = self.data.Close[-1]
            stop_loss = max(self.price_highs[-1] * 1.02, entry_price * 1.03)
            risk_per_share = stop_loss - entry_price
            
            if risk_per_share <= 0:
                return  # Avoid invalid stop
            
            position_size = (self.equity * self.risk_per_trade) / risk_per_share
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.sell(size=position_size, sl=stop_loss, 
                        tag="🌙 BEARISH DIVERGENCE SHORT")
                print(f"🚀 ENTRY: {position_size} shorts @ {entry_price:.2f} | SL: {stop_loss:.2f}")
                
                # 📊 Track exit parameters
                self.entry_volume = self.data.Volume[-1]
                self.entry_atr = self.atr[-1]
                self.entry_bar = len(self.data)

        # 💸 EXIT Conditions
        if self.position:
            # 📉 Volume Drop Exit
            if self.data.Volume[-1] < self.entry_volume * 0.7:
                self.position.close()
                print(f"📉 EXIT: Volume drop @ {self.data.Close[-1]:.2f}")

            # ⚡ Volatility Spike Exit
            elif self.atr[-1] > self.entry_atr * 1.10:
                self.position.close()
                print(f"⚡ EXIT: Volatility spike @ {self.data