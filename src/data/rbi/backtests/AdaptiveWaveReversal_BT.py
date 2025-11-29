Below is an example implementation of AdaptiveWaveReversal using backtesting.py with TA‑lib (and a bit of pandas for custom calculations). Run the script (for example in a .py file) to read the data, clean it, set up the indicators with self.I(), and then print full stats. Enjoy the Moon Dev magic! 🌙✨🚀

--------------------------------------------------
#!/usr/bin/env python3
"""
Moon Dev’s Backtest AI 🌙
Strategy: AdaptiveWaveReversal
--------------------------------------------------
Instructions:
• Cleans the CSV column names and drops unnamed columns.
• Uses TA‑lib for indicator calculations via self.I().
• Implements long/short entries based on a higher timeframe “trend” (approximated using VWAP),
  swing highs/lows (via talib.MAX/min), and pivot points for dynamic support/resistance.
• Uses risk management (1% risk per trade on a 1,000,000 account) and proper integer position sizing.
• Prints themed debug messages.
• Finally, prints full backtest stats.
--------------------------------------------------
NOTE: No charting is invoked. Only stats are printed.
"""

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ***********************************************
# Helper functions using TA-lib / pandas for indicators
# ***********************************************

def calc_vwap(close, high, low, volume):
    # VWAP computation using typical price and cumulative sum.
    # Typical Price = (High+Low+Close)/3
    typical_price = (high + low + close) / 3.0
    cumulative_tp_volume = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    vwap = cumulative_tp_volume / cumulative_volume
    return vwap

def calc_pivot(high, low, close):
    # Standard pivot point: (High + Low + Close)/3
    return (high + low + close) / 3.0

# ***********************************************
# Strategy Class Definition
# ***********************************************

class AdaptiveWaveReversal(Strategy):
    # Set the default cash size to 1,000,000
    cash = 1_000_000

    def init(self):
        # *******************************
        # Compute Indicators via self.I wrapper
        # *******************************
        # Compute Swing High (rolling maximum) and Swing Low (rolling minimum) over a 20-bar window
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low  = self.I(talib.MIN,  self.data.Low,  timeperiod=20)

        # Compute VWAP using our custom function - note: VWAP is not available in TA‑lib so we use our function.
        self.vwap = self.I(calc_vwap, self.data.Close, self.data.High, self.data.Low, self.data.Volume)
        
        # Compute Pivot value using our custom pivot function (for an extra dynamic level)
        self.pivot = self.I(calc_pivot, self.data.High, self.data.Low, self.data.Close)

        # Debug print for initialization
        print("🌙🚀 [Moon Dev] Strategy Initialized with indicators!")

    def next(self):
        price = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        current_pivot = self.pivot[-1]
        # Using previous swing values for entry thresholds (note: index -2 because current bar might not have closed fully)
        prev_swing_high = self.swing_high[-2] if len(self.swing_high) > 1 else None
        prev_swing_low = self.swing_low[-2] if len(self.swing_low) > 1 else None

        # Determine trend direction using VWAP
        if price > current_vwap:
            trend = "uptrend"
        else:
            trend = "downtrend"
        print(f"🌙✨ [Moon Dev] Bar: {self.data.index[-1]} | Price: {price:.2f} | VWAP: {current_vwap:.2f} | Trend: {trend}")

        # Risk management parameters
        # 1% risk per trade on our equity.
        risk_percent = 0.01  
        account_risk = self.equity * risk_percent

        # --------------------------
        # Entry Logic:
        # --------------------------
        # If we are flat (no open positions), try to enter a trade.
        if not self.position:
            # LONG Entry: For an uptrend and breakout above previous swing high.
            if trend == "uptrend" and prev_swing_high is not None and price > prev_swing_high:
                # For long, set stop loss just below the recent swing low (if available; else fallback to pivot- a bit lower)
                stop_loss = self.swing_low[-1] if self.swing_low[-1] < price else current_pivot * 0.995
                risk_per_unit = price - stop_loss
                if risk_per_unit <= 0:   # Protect against divide-by-zero or negative risk
                    print("🌙🚀 [Moon Dev] Invalid risk calculation for LONG - skipping entry.")
                else:
                    pos_size = int(round(account_risk / risk_per_unit))
                    if pos_size <= 0:
                        print("🌙🚀 [Moon Dev] Calculated LONG position size is zero - skipping entry.")
                    else:
                        # Set fixed target: profit target 10 points above entry (could be adapted)
                        take_profit = price + 10  
                        print(f"🌙🚀 [Moon Dev] LONG SIGNAL! Price: {price:.2f}, Stop Loss: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: {pos_size}")
                        self.buy(size=pos_size, sl=stop_loss, tp=take_profit)

            # SHORT Entry: For a downtrend and breakout below previous swing low.
            elif trend == "downtrend" and prev_swing_low is not None and price < prev_swing_low:
                # For short, set stop loss just above the recent swing high (or use pivot+ a bit higher)
                stop_loss = self.swing_high[-1] if self.swing_high[-1] > price else current_pivot * 1.005
                risk_per_unit = stop_loss - price
                if risk_per_unit <= 0:
                    print("🌙🚀 [Moon Dev] Invalid risk calculation for SHORT - skipping entry.")
                else:
                    pos_size = int(round(account_risk / risk_per_unit))
                    if pos_size <= 0:
                        print("🌙🚀 [Moon Dev] Calculated SHORT position size is zero - skipping entry.")
                    else:
                        take_profit = price - 10  
                        print(f"🌙🚀 [Moon Dev] SHORT SIGNAL! Price: {price:.2f}, Stop Loss: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: {pos_size}")
                        self.sell(size=pos_size, sl=stop_loss, tp=take_profit)

        # --------------------------
        # Exit Logic:
        # --------------------------
        # For open LONG positions: exit if price falls below VWAP (potential trend reversal)
        if self.position and self.position.is_long:
            if price < current_vwap:
                print(f"🌙⛔ [Moon Dev] Exiting LONG position as price dropped below VWAP. Price: {price:.2f}, VWAP: {current_vwap:.2f}")
                self.position.close()
        # For open SHORT positions: exit if price rises above VWAP
        elif self.position and self.position.is_short:
            if price > current_vwap:
                print(f"🌙⛔ [Moon Dev] Exiting SHORT position as price rose above VWAP. Price: {price:.2f}, VWAP: {current_vwap:.2f}")
                self.position.close()

        # (Optional) Trailing stop logic can be added here.
        # For now, the initial SL set on entry will do the job – you may update self.position.sl manually if desired.

# ***********************************************
# Data Loading and Cleaning
# ***********************************************

def load_and_clean_data(filepath):
    # Read the CSV file
    data = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    # Clean column names: remove spaces and lower-case them.
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns (if any)
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Rename required columns to proper backtesting.py requirements (capital first letters)
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=rename_map, inplace=True)
    print("🌙✨ [Moon Dev] Data loaded and cleaned!")
    return data

# ***********************************************
# Main: Run Backtest
# ***********************************************

if __name__ == '__main__':
    DATA_PATH = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    data = load_and_clean_data(DATA_PATH)

    # Create Backtest object and run using default parameters first.
    bt = Backtest(data, AdaptiveWaveReversal, cash=1_000_000, commission=.000,
                  exclusive_orders=True)
    stats = bt.run()
    
    # Print full stats and strategy details
    print("\n🌙🚀 [Moon Dev] FULL STATS:")
    print(stats)
    print("\n🌙🚀 [Moon Dev] STRATEGY DETAILS:")
    print(stats._strategy)

# End of Script
--------------------------------------------------

Notes:
• We compute all custom indicators (VWAP, Pivot, Swing High/Low) using self.I().
• Entry conditions check the trend (approximated by VWAP) and a breakout above/below recent swing highs/lows.
• Risk management uses 1% of equity per trade with proper integer position sizing.
• Plenty of Moon Dev-themed debug prints are added for clarity. 

Happy backtesting! 🌙✨🚀