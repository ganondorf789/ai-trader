"""
Lingrid Rising Channel Strategy Backtest
Based on: https://www.tradingview.com/chart/XAUUSD/6q77Of5I-Lingrid-GOLD-Sideways-Movement-Ahead-of-FOMC-Decision/

Strategy Logic:
- Identify rising channel using linear regression
- Buy at channel support (lower band) when higher lows form
- Sell at channel resistance (upper band)
- Stop loss on channel breakdown

Uses backtesting.py with pandas_ta for indicators
"""

import os
import glob
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy import stats


def calculate_linear_regression_channel(data, period=20, deviations=2.0):
    """Calculate linear regression channel bands"""
    close = data['Close'].values
    n = len(close)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)

    for i in range(period - 1, n):
        y = close[i - period + 1:i + 1]
        x = np.arange(period)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        predicted = slope * (period - 1) + intercept
        residuals = y - (slope * x + intercept)
        std_dev = np.std(residuals)

        middle[i] = predicted
        upper[i] = predicted + deviations * std_dev
        lower[i] = predicted - deviations * std_dev
        slope_arr[i] = slope

    return pd.Series(upper, index=data.index), pd.Series(middle, index=data.index), \
           pd.Series(lower, index=data.index), pd.Series(slope_arr, index=data.index)


def detect_higher_lows(data, lookback=5):
    """Detect higher lows pattern for bullish structure confirmation"""
    low = data['Low'].values
    n = len(low)
    higher_lows = np.zeros(n)

    for i in range(lookback * 2, n):
        # Find recent swing lows
        recent_low = np.min(low[i - lookback:i])
        prev_low = np.min(low[i - lookback * 2:i - lookback])

        if recent_low > prev_low:
            higher_lows[i] = 1

    return pd.Series(higher_lows, index=data.index)


class LingridChannelStrategy(Strategy):
    """
    Rising Channel Strategy with Higher Lows Confirmation

    Parameters:
    - channel_period: Period for linear regression channel calculation
    - channel_deviations: Standard deviations for channel width
    - hl_lookback: Lookback period for higher lows detection
    - atr_period: ATR period for stop loss calculation
    - risk_reward: Risk/reward ratio for take profit
    - trailing_stop_atr: ATR multiplier for trailing stop
    """

    channel_period = 20
    channel_deviations = 2.0
    hl_lookback = 5
    atr_period = 14
    risk_reward = 2.0
    trailing_stop_atr = 2.0

    def init(self):
        # Calculate linear regression channel
        self.upper, self.middle, self.lower, self.slope = \
            self.I(lambda: calculate_linear_regression_channel(
                self.data.df, self.channel_period, self.channel_deviations
            ), plot=False)

        # Higher lows detection
        self.higher_lows = self.I(lambda: detect_higher_lows(
            self.data.df, self.hl_lookback
        ), plot=False)

        # ATR for stop loss
        self.atr = self.I(ta.atr,
                          pd.Series(self.data.High),
                          pd.Series(self.data.Low),
                          pd.Series(self.data.Close),
                          self.atr_period)

        # RSI for momentum confirmation
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)

        # Track entry price for trailing stop
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.highest_since_entry = None

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.channel_period + 10:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(self.lower[-1]) or np.isnan(atr) or atr == 0:
            return

        upper = self.upper[-1]
        lower = self.lower[-1]
        middle = self.middle[-1]
        slope = self.slope[-1]
        higher_lows = self.higher_lows[-1]
        rsi = self.rsi[-1] if not np.isnan(self.rsi[-1]) else 50

        # Position management
        if self.position:
            # Update trailing stop for long positions
            if self.position.is_long:
                if self.highest_since_entry is None:
                    self.highest_since_entry = price
                else:
                    self.highest_since_entry = max(self.highest_since_entry, price)

                # Trailing stop based on ATR
                trailing_stop = self.highest_since_entry - self.trailing_stop_atr * atr

                # Exit conditions for long
                # 1. Price reaches upper channel (take profit zone)
                # 2. Channel breakdown (slope turns negative)
                # 3. Trailing stop hit
                if price >= upper * 0.98:  # Near upper channel
                    self.position.close()
                    self.reset_position_vars()
                elif slope < 0 and price < lower:  # Channel breakdown
                    self.position.close()
                    self.reset_position_vars()
                elif price < trailing_stop:  # Trailing stop
                    self.position.close()
                    self.reset_position_vars()

            # Update trailing stop for short positions
            elif self.position.is_short:
                if self.highest_since_entry is None:
                    self.highest_since_entry = price
                else:
                    self.highest_since_entry = min(self.highest_since_entry, price)

                trailing_stop = self.highest_since_entry + self.trailing_stop_atr * atr

                # Exit conditions for short
                if price <= lower * 1.02:  # Near lower channel
                    self.position.close()
                    self.reset_position_vars()
                elif slope > 0 and price > upper:  # Channel breakout
                    self.position.close()
                    self.reset_position_vars()
                elif price > trailing_stop:
                    self.position.close()
                    self.reset_position_vars()

        else:
            # Entry logic
            # Long entry: Price near lower channel + rising channel + higher lows + RSI not overbought
            near_support = price <= lower * 1.02
            rising_channel = slope > 0
            bullish_structure = higher_lows == 1
            momentum_ok = rsi < 70

            if near_support and rising_channel and bullish_structure and momentum_ok:
                self.entry_price = price
                self.stop_loss = lower - 1.5 * atr
                self.take_profit = upper
                self.highest_since_entry = price
                self.buy()

            # Short entry: Price near upper channel + falling channel + RSI overbought
            near_resistance = price >= upper * 0.98
            falling_channel = slope < 0
            momentum_bearish = rsi > 70

            if near_resistance and falling_channel and momentum_bearish:
                self.entry_price = price
                self.stop_loss = upper + 1.5 * atr
                self.take_profit = lower
                self.highest_since_entry = price
                self.sell()

    def reset_position_vars(self):
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.highest_since_entry = None


def load_and_prepare_data(file_path):
    """Load CSV and prepare for backtesting.py format"""
    df = pd.read_csv(file_path)

    # Rename columns to match backtesting.py requirements
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'timestamp': 'Datetime'
    })

    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Select only required columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Remove rows with zero or invalid data
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]

    # Sort by datetime
    df = df.sort_index()

    return df


def run_backtest(file_path, cash=100000, commission=0.001):
    """Run backtest on a single file"""
    print(f"\n{'='*60}")
    print(f"Backtesting: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        df = load_and_prepare_data(file_path)

        if len(df) < 50:
            print(f"  Skipping: Not enough data ({len(df)} rows)")
            return None

        print(f"  Data range: {df.index[0]} to {df.index[-1]}")
        print(f"  Total bars: {len(df)}")
        print(f"  Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

        bt = Backtest(
            df,
            LingridChannelStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True
        )

        stats = bt.run()

        return stats

    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def format_stats(stats):
    """Format backtest statistics for display"""
    if stats is None:
        return "No results"

    output = []
    output.append(f"  Return [%]:              {stats['Return [%]']:.2f}%")
    output.append(f"  Buy & Hold Return [%]:   {stats['Buy & Hold Return [%]']:.2f}%")
    output.append(f"  Max Drawdown [%]:        {stats['Max. Drawdown [%]']:.2f}%")
    output.append(f"  # Trades:                {stats['# Trades']}")

    if stats['# Trades'] > 0:
        output.append(f"  Win Rate [%]:            {stats['Win Rate [%]']:.2f}%")
        output.append(f"  Profit Factor:           {stats['Profit Factor']:.2f}" if not np.isnan(stats['Profit Factor']) else "  Profit Factor:           N/A")
        output.append(f"  Sharpe Ratio:            {stats['Sharpe Ratio']:.2f}" if not np.isnan(stats['Sharpe Ratio']) else "  Sharpe Ratio:            N/A")
        output.append(f"  Sortino Ratio:           {stats['Sortino Ratio']:.2f}" if not np.isnan(stats['Sortino Ratio']) else "  Sortino Ratio:            N/A")
        output.append(f"  Avg Trade [%]:           {stats['Avg. Trade [%]']:.2f}%")
        output.append(f"  Best Trade [%]:          {stats['Best Trade [%]']:.2f}%")
        output.append(f"  Worst Trade [%]:         {stats['Worst Trade [%]']:.2f}%")

    output.append(f"  Final Equity:            ${stats['Equity Final [$]']:,.2f}")

    return "\n".join(output)


def main():
    """Main function to run backtests on all OHLCV files"""
    print("="*60)
    print("Lingrid Rising Channel Strategy Backtest")
    print("Based on TradingView analysis")
    print("="*60)

    # Find all OHLCV files
    ohlcv_dir = os.path.join(os.path.dirname(__file__), "..", "ohlcv")
    ohlcv_files = glob.glob(os.path.join(ohlcv_dir, "*.csv"))

    if not ohlcv_files:
        print(f"No CSV files found in {ohlcv_dir}")
        return

    print(f"\nFound {len(ohlcv_files)} data file(s)")

    all_results = []

    for file_path in ohlcv_files:
        stats = run_backtest(file_path)

        if stats is not None:
            print("\n  Results:")
            print(format_stats(stats))

            all_results.append({
                'file': os.path.basename(file_path),
                'return': stats['Return [%]'],
                'trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if stats['# Trades'] > 0 else 0,
                'max_dd': stats['Max. Drawdown [%]'],
                'sharpe': stats['Sharpe Ratio'] if not np.isnan(stats['Sharpe Ratio']) else 0,
                'final_equity': stats['Equity Final [$]']
            })

    # Summary
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        df_results = pd.DataFrame(all_results)

        print(f"\nTotal files tested: {len(df_results)}")
        print(f"Average Return: {df_results['return'].mean():.2f}%")
        print(f"Total Trades: {df_results['trades'].sum()}")
        print(f"Average Win Rate: {df_results['win_rate'].mean():.2f}%")
        print(f"Average Max Drawdown: {df_results['max_dd'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {df_results['sharpe'].mean():.2f}")

        print("\n" + "-"*60)
        print("Per-file Summary:")
        print("-"*60)
        for _, row in df_results.iterrows():
            print(f"  {row['file']:<25} Return: {row['return']:>8.2f}%  Trades: {row['trades']:>3}  Win: {row['win_rate']:>5.1f}%")

    return all_results


if __name__ == "__main__":
    results = main()
