import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker as plticker
from matplotlib import ticker
import seaborn as sns
from scipy.stats import zscore

def dataframe_info(df):
    report = pd.DataFrame(columns=['Column', 'Data Type', 'Unique Count', 'Unique Sample', 'Missing Values', 'Missing Percentage'])
    for column in df.columns:
        data_type = df[column].dtype
        unique_count = df[column].nunique()
        unique_sample = df[column].unique()[:5]
        missing_values = df[column].isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        report = pd.concat([report, pd.DataFrame({'Column': [column],
                                                      'Data Type': [data_type],
                                                      'Unique Count': [unique_count],
                                                      'Unique Sample': [unique_sample],
                                                      'Missing Values': [missing_values],
                                                      'Missing Percentage': [missing_percentage.round(4)]})],
                             ignore_index=True)
    return report

def plot_sparsity(df, col, target_col):
    stats = df[[col, target_col]].groupby(col).agg(['count', 'mean', 'sum'])
    stats = stats.reset_index()
    stats.columns = [col, 'count', 'mean', 'sum']
    stats_sort = stats['count'].value_counts().reset_index()
    stats_sort = stats_sort.sort_values('index')
    plt.figure(figsize=(15,4))
    plt.bar(stats_sort['index'].astype(str).values[0:20], stats_sort['count'].values[0:20])
    plt.title('Frequency of ' + str(col))
    plt.xlabel('Number frequency')
    plt.ylabel('Frequency')

def plot_top20(df, col, target_col):
    stats = df[[col, target_col]].groupby(col).agg(['count', 'mean', 'sum'])
    stats = stats.reset_index()
    stats.columns = [col, 'count', 'mean', 'sum']
    stats = stats.sort_values('count', ascending=False)
    fig, ax1 = plt.subplots(figsize=(15,4))
    ax2 = ax1.twinx()
    ax1.bar(stats[col].astype(str).values[0:20], stats['count'].values[0:20])
    ax1.set_xticklabels(stats[col].astype(str).values[0:20], rotation='vertical')
    ax2.plot(stats['mean'].values[0:20], color='red')
    ax2.set_ylim(0,1)
    ax2.set_ylabel('Mean Target')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel(col)
    ax1.set_title('Top20 ' + col + 's based on frequency')

def visualize_category_counts(df, column, ax):
    # Count the occurrences of each unique value
    value_counts = df[column].value_counts()

    # Calculate the percentage of each value
    percentages = value_counts / len(df) * 100

    # Sort the values in descending order
    sorted_values = value_counts.sort_values(ascending=False)

    if len(sorted_values) > 2:
        # If non-binary, use seaborn barplot
        sns.barplot(x=sorted_values.values, y=sorted_values.index, ax=ax, palette='copper_r')
        ax.set_title(f'{column} Counts')
        ax.set_xlabel('Count')
        ax.set_ylabel(column)

        # Add count and percentage text to each bar
        for i, count in enumerate(sorted_values):
            percentage = percentages[sorted_values.index[i]]
            ax.text(count, i, f"{count} ({percentage:.1f}%)", va='center')
    else:
        # If binary, use a pie chart
        ax.pie(sorted_values, labels=sorted_values.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{column} Distribution')

    ax.grid(False)


def analyze_missing(df, min_percent_missing=0, max_percent_missing=100, plot_distribution=False):
    # calculate the number of missing values for each column
    missing_counts = df.isnull().sum()
    # create a new DataFrame with the variable name and number of missing values for each column
    output_df = pd.DataFrame(data=missing_counts.values, index=missing_counts.index, columns=['Missing Values'])
    # calculate the percentage of missing values for each column
    output_df['Percent Missing'] = output_df['Missing Values'] / df.shape[0] * 100
    # sort the output DataFrame by the number of missing values in descending order
    output_df = output_df.sort_values(by='Missing Values', ascending=False)

    if plot_distribution:
        # plot a histogram of the percentage of missing values for all columns
        plt.hist(output_df['Percent Missing'], bins=20, edgecolor='black')
        plt.title('Distribution of Missing Values')
        plt.xlabel('Percent Missing')
        plt.ylabel('Count')
        plt.show()

    # only return columns where the percentage of missing values is within the range [min_percent_missing, max_percent_missing]
    output_df = output_df[(output_df['Percent Missing'] >= min_percent_missing) & (output_df['Percent Missing'] <= max_percent_missing)]

    return output_df

def count_rows_with_blankspace(df):
    """
    This function takes a Pandas DataFrame as input and returns a new DataFrame with the number of rows
    having a blankspace as value for each column.
    """
    missing_counts = df.astype(str).apply(lambda x: x.str.isspace().sum())
    missing_df = pd.DataFrame({'Missing Rows': missing_counts}).sort_values(by='Missing Rows', ascending=False)
    return missing_df

def replace_blankspace_with_nan(df):
    """
    This function takes a Pandas DataFrame as input and replaces any rows with blank spaces with NaN values.
    """
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df


def find_outlier_columns(df, method='z-score', threshold=3, factor=1.5):
    outlier_cols = []
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Exclude binary columns
    num_cols = [col for col in num_cols if df[col].nunique() > 2]

    if method == 'z-score':
        for col in num_cols:
            z_scores = zscore(df[col].dropna())
            if (np.abs(z_scores) > threshold).any():
                outlier_cols.append(col)
    elif method == 'iqr':
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if ((df[col] < (Q1 - factor * IQR)) | (df[col] > (Q3 + factor * IQR))).any():
                outlier_cols.append(col)
    else:
        print(f"Unsupported method: {method}")
        return None

    return outlier_cols


def detect_outliers(df, column, plot=False):
    """
    This function returns a DataFrame that consists of outliers in the specified column of the input DataFrame.
    An optional boxplot for visualizing the outliers can be displayed if plot=True.

    Args:
    df (pandas.DataFrame): The input DataFrame.
    column (str): The name of the column in which to search for outliers.
    plot (bool): If True, display a boxplot for the specified column. Default is False.

    Returns:
    outliers (pandas.DataFrame): A DataFrame that consists of outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    if plot:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=df[column])
        plt.title('Boxplot for ' + column)
        plt.show()

    return outliers

def discretize(v, v_intervals, use_quartiles=False, use_continuous_bins=False):
    if isinstance(v, (pd.core.series.Series, np.ndarray)) and isinstance(v_intervals, (list, np.ndarray)) and len(np.unique(v)) != len(v_intervals):
        raise Exception("length of interval must match unique items in array")

    if isinstance(v, (str)) and isinstance(v_intervals, (list, np.ndarray)):
        #name of variable instead of array and list of intervals used
        if isinstance(v_intervals, list): v_intervals = np.array(v_intervals)
        return v, v_intervals

    if (np.isin(v.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32'])) and (isinstance(v_intervals, (int))) and (len(np.unique(v)) >= v_intervals) and (max(v) > min(v)):
        #v is discretizable, otherwise assumed to be already discretized
        if use_continuous_bins:
            if use_quartiles:
                v, bins = pd.qcut(v, v_intervals, duplicates='drop', retbins=True, labels=True, precision=2)
            else:
                v, bins = pd.cut(v, v_intervals, duplicates='drop', retbins=True, labels=True, precision=2)
        else:
            if use_quartiles:
                v = pd.qcut(v, v_intervals, duplicates='drop', precision=2)
            else:
                v = pd.cut(v, v_intervals, duplicates='drop', precision=2)

    if np.isin(v.dtype, [object, 'category']):
        if not isinstance(v, (pd.core.series.Series)):
            v = pd.Series(v)
        bins = np.sort(np.unique(v)).astype(str)
        v = v.astype(str)
        bin_dict = {bins[i]:i for i in range(len(bins))}
        v = v.replace(bin_dict)
    else:
        bins = np.unique(v)

    if isinstance(v_intervals, (list, np.ndarray)) and len(bins) == len(v_intervals):
        bins = v_intervals

    return v, bins

def plot_prob_progression(x, y, x_intervals=7, use_quartiles=False,\
                          xlabel=None, ylabel=None, title=None, text=None, model=None, X_df=None, x_col=None,\
                         mean_line=False, figsize=(12,6), x_margin=0.01, color='Reds'):
    x = x.astype(int)
    y = y.astype(int)
    if isinstance(x, list): x = np.array(x)
    if isinstance(y, list): y = np.array(y)
    if (not isinstance(x, (str, pd.core.series.Series, np.ndarray))) or (not isinstance(y, (str, pd.core.series.Series, np.ndarray))):
        raise Exception("x and y must be either lists, pandas series or numpy arrays. x can be string when dataset is provided seperately")
    if (isinstance(x, (pd.core.series.Series, np.ndarray)) and (len(x.shape) != 1)) or ((isinstance(y, (pd.core.series.Series, np.ndarray))) and (len(y.shape) != 1)):
        raise Exception("x and y must have a single dimension")
    if (isinstance(x_intervals, (int)) and (x_intervals < 2)) or (isinstance(x_intervals, (list, np.ndarray)) and (len(x_intervals) < 2)):
        raise Exception("there must be at least two intervals to plot")
    if not np.isin(y.dtype, [int, float, 'int8', 'int16', 'int32', 'float16', 'float32']):
        raise Exception("y dimension must be a list, pandas series or numpy array of integers or floats")
    if max(y) == min(y):
        raise Exception("y dimension must have at least two values")
    elif len(np.unique(y)) == 2 and ((max(y) != 1) or (min(y) != 0)):
        raise Exception("y dimension if has two values must have a max of exactly 1 and min of exactly zero")
    elif len(np.unique(y)) > 2 and ((max(y) <= 1) or (min(y) >= 0)):
        raise Exception("y dimension if has more than two values must have range between between 0-1")
    x_use_continuous_bins = (model is not None) and (isinstance(x_intervals, (list, np.ndarray)))
    x, x_bins = discretize(x, x_intervals, use_quartiles, x_use_continuous_bins)
    x_range = [*range(len(x_bins))]
    plot_df = pd.DataFrame({'x':x_range})
    if (model is not None) and (X_df is not None) and (x_col is not None):
        preds = model.predict(X_df).squeeze()
        if len(np.unique(preds)) <= 2:
            preds = model.predict_proba(X_df)[:,1]
        x_, _ = discretize(X_df[x_col], x_intervals, use_quartiles, x_use_continuous_bins)
        xy_df = pd.DataFrame({'x':x_, 'y':preds})
    else:
        xy_df = pd.DataFrame({'x':x,'y':y})
    probs_df = xy_df.groupby(['x']).mean().reset_index()
    probs_df = pd.merge(plot_df, probs_df, how='left', on='x').fillna(0)

    x_bin_cnt = len(x_bins)
    l_width = 0.933
    r_width = 0.05
    w, h = figsize
    wp = (w-l_width-r_width)/9.27356902357
    xh_margin = ((wp-(x_margin*2))/(x_bin_cnt*2))+x_margin
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize,\
                                   gridspec_kw={'height_ratios': [3, 1]})
    if title is not None:
        fig.suptitle(title, fontsize=21)
        ax0.text(0.61, 0.85, text,
                 horizontalalignment='left', verticalalignment='top', transform=ax0.transAxes, fontsize=9, fontstyle='italic')
        plt.subplots_adjust(top = 0.92, bottom=0.01, hspace=0.001, wspace=0.001)
    else:
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.001, wspace=0.001)

    ax0.minorticks_on()
    # Disable grid for ax0
    ax0.grid(False)
    cmap = mpl.colormaps[color]
    num_segments = len(probs_df['y']) - 1

    for i in range(num_segments):
        segment = probs_df.iloc[i:i+2]
        color = cmap(i / num_segments)
        sns.lineplot(data=segment, x='x', y='y', marker='o', color=color, ax=ax0)

    # sns.lineplot(data=probs_df, x='x', y='y', marker='o', ax=ax0)
    ax0.set_ylabel('Probability', fontsize=15)
    ax0.set_xlabel('')

    if mean_line:
        ax0.axhline(y=xy_df.y.mean(), c='#E9EAE5', alpha=0.6, linestyle='dotted', label="mean")
        ax0.legend()

    
    colors = [cmap(i) for i in np.linspace(0, 1, len(x_bins))]

    # Disable grid for ax1
    ax1.grid(False)

    hist = sns.histplot(xy_df, x="x", stat='probability', bins=np.arange(x_bin_cnt+1)-0.5, ax=ax1)
    # color the bars using the color map
    for patch, color in zip(hist.patches, colors):
        patch.set_facecolor(color) # color depends on the index of the bar
    ax1.set_ylabel('Observations', fontsize=15)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax1.set_xticklabels(['']+['(' + str(round(float(i.split(',')[0][1:]))) + ', ' + str(round(float(i.split(',')[1][:-1]))) + ']' for i in x_bins])
    ax1.margins(x=x_margin)
    plt.show()
    
def plot_dist_kurtosis(df, numerical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(10,5))
        sns.distplot(df[col].dropna(), kde=True)

        kurtosis = df[col].kurtosis()

        # Define kurtosis category
        if kurtosis > 0:
            kurtosis_category = 'Leptokurtic'
        elif kurtosis < 0:
            kurtosis_category = 'Platykurtic'
        else:
            kurtosis_category = 'Mesokurtic'

        # Annotate the kurtosis category
        plt.text(0.97, 0.97, f'Kurtosis: {kurtosis:.2f}\n{kurtosis_category}',
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes)

        plt.title(f'Distribution of {col} (kurtosis: {kurtosis:.2f})')
        plt.show()