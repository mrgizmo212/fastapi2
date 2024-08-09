import pandas as pd
import numpy as np

def detect_head_shoulder(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_head_shoulder = ((df['high_roll_max'] > df['High'].shift(1)) & (df['high_roll_max'] > df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low'].shift(1)) & (df['low_roll_min'] < df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)))
    df['head_shoulder_pattern'] = ''
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    patterns_with_times = {
        'Head and Shoulder': df.index[mask_head_shoulder].tolist(),
        'Inverse Head and Shoulder': df.index[mask_inv_head_shoulder].tolist()
    }
    return df, patterns_with_times

def detect_multiple_tops_bottoms(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    mask_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['close_roll_max'] < df['Close'].shift(1))
    mask_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['close_roll_min'] > df['Close'].shift(1))
    df['multiple_top_bottom_pattern'] = ''
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    
    patterns_with_times = {
        'Multiple Top': df.index[mask_top].tolist(),
        'Multiple Bottom': df.index[mask_bottom].tolist()
    }
    return df, patterns_with_times

def calculate_support_resistance(df, window=3):
    roll_window = window
    std_dev = 2
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()
    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()
    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    return df, {}  # No specific patterns to return times for

def detect_triangle_pattern(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    df['triangle_pattern'] = ''
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    
    patterns_with_times = {
        'Ascending Triangle': df.index[mask_asc].tolist(),
        'Descending Triangle': df.index[mask_desc].tolist()
    }
    return df, patterns_with_times

def detect_wedge(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['wedge_pattern'] = ''
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    
    patterns_with_times = {
        'Wedge Up': df.index[mask_wedge_up].tolist(),
        'Wedge Down': df.index[mask_wedge_down].tolist()
    }
    return df, patterns_with_times

def detect_channel(df, window=3):
    roll_window = window
    channel_range = 0.1
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['channel_pattern'] = ''
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    
    patterns_with_times = {
        'Channel Up': df.index[mask_channel_up].tolist(),
        'Channel Down': df.index[mask_channel_down].tolist()
    }
    return df, patterns_with_times

def detect_double_top_bottom(df, window=3, threshold=0.05):
    roll_window = window
    range_threshold = threshold
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_double_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['high_roll_max'] >= df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    mask_double_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    df['double_pattern'] = ''
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'
    
    patterns_with_times = {
        'Double Top': df.index[mask_double_top].tolist(),
        'Double Bottom': df.index[mask_double_bottom].tolist()
    }
    return df, patterns_with_times

def detect_trendline(df, window=2):
    roll_window = window
    df['slope'] = np.nan
    df['intercept'] = np.nan
    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['Close'].iloc[i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c
    mask_support = df['slope'] > 0
    mask_resistance = df['slope'] < 0
    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']
    
    patterns_with_times = {
        'Support Trendline': df.index[mask_support].tolist(),
        'Resistance Trendline': df.index[mask_resistance].tolist()
    }
    return df, patterns_with_times

def find_pivots(df):
    high_diffs = df['high'].diff()
    low_diffs = df['low'].diff()
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)
    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'
    
    patterns_with_times = {
        'Higher High': df.index[higher_high_mask].tolist(),
        'Lower Low': df.index[lower_low_mask].tolist(),
        'Lower High': df.index[lower_high_mask].tolist(),
        'Higher Low': df.index[higher_low_mask].tolist()
    }
    return df, patterns_with_times
