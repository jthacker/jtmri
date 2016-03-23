import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
from collections import namedtuple
import logging


log = logging.getLogger(__name__)


def cohens_d(a, b):
    '''Cohen's D computed between two list like sequences''' 
    return (np.mean(a) - np.mean(b)) / ((np.var(a) + np.var(b)/2.0))**0.5

def sig(p):
    ret = ''
    if p < 0.01:
        ret = '**'
    elif p < 0.05:
        ret = '*'
    return ret

def fwer_sequential_adjust(df, pval_col='p'):
    '''Familywise Error Rate adjustment''' 
    pvals = df[pval_col].order()
    return np.arange(len(pvals), 0, -1) * pvals


class Tests(object):
    def __init__(self, a_name, a_filter, b_name, b_filter, cols):
        self.a_name = a_name
        self.a_filter = a_filter
        self.b_name = b_name
        self.b_filter = b_filter
        self.cols = cols
   

def tests(data, groups, tests, stat, adj=None):
    """Generate a table of test statistics.
    Args:
        data   -- Pandas dataframe
        groups -- Groups
        tests  -- Tests object
        stat   -- Statistics to perform for each test
        adj    -- (default: None) Adjust p-values byt groups if True
    """
    out = []
    cols = ['test']
    cols += list(groups)
    cols += ['mean(%s)' % tests.a_name, 'std(%s)' % tests.a_name]
    cols += ['mean(%s)' % tests.b_name, 'std(%s)' % tests.b_name]
    cols += ['N(%s)' % tests.a_name, 'N(%s)' % tests.b_name]
    cols += ['p', 'p_sig', 'd']
   
    for key, grp_idx in data.groupby(groups).groups.items():
        df = data.ix[grp_idx]
        df_a = tests.a_filter(df)
        df_b = tests.b_filter(df)
        for col in tests.cols:
            a, b = df_a[col], df_b[col]
            statistic, p = stat(a, b)
            row = [col]
            row += [key]
            row += [a.mean(), a.std(), b.mean(), b.std()]
            row += [len(a), len(b)]
            row += [p, sig(p), cohens_d(a,b)]
            out.append(row)
            
    df = pd.DataFrame(out, columns=cols)
    if adj:
        df['padj'] = np.NaN * np.ones(len(df))
        for key, grp_idx in df.groupby(groups).groups.items():
            padj = fwer_sequential_adjust(df.ix[grp_idx])
            df.loc[padj.index, 'padj'] = padj
        df['padj_sig'] = df['padj'].apply(sig)
    
    df = df.set_index(['test'] + list(groups))
    df = df.sortlevel()
    return df


def test_grid(data, test_groups, data_columns, statistic, column_groups=None, adjust=False):
    """Generate a table of test statistics.
    Parameters
    ----------
    data : DataFrame
        Pandas dataframe to perform tests on
    test_groups : 
        An iterable where each item contains: column name, filter function
        The number of test columns will depend on the statistic being computed.
        For a T-test, it should be two, since only two groups of data are being
        compared.
    data_columns : list of column names
        A list of column names to perform each the test on
    statistic : function that returns p value
        Statistic to compute
    column_groups : Iterable of column names (default: None)
        Column names to group on before performing the tests
    adjust : bool (default: False)
        Perform groupwise adjustment of p-values for each of the column groups

    Returns
    -------
    DataFrame
    """
    summary_stats = (
        ('mean', lambda arr: arr.mean()),
        ('std', lambda arr: arr.std()))

    out = []
    col_names = ['test']
    if column_groups is not None:
        col_names += list(column_groups)
    for name, _ in test_groups:
        col_names += ['{}({})'.format(stat_name, name) for stat_name, _ in summary_stats]
    for name, _ in test_groups:
        col_names += ['N({})'.format(name)]
    col_names += ['p', 'sig', 'd']

    groups = True
    if column_groups is None:
        groups = False
        column_groups = lambda x: 0
        column_group_names = []

    for key, grp_idx in data.groupby(column_groups).groups.items():
        df = data.ix[grp_idx]
        test_dfs = [f(df) for _, f in test_groups]
        for data_col in data_columns:
            test_arrays = [tdf[data_col] for tdf in test_dfs]
            _, p = statistic(*test_arrays)
            row = [data_col]
            if groups:
                row += list(key)
            # Compute summary statistics
            for arr in test_arrays:
                row += [func(arr) for _, func in summary_stats]
            for arr in test_arrays:
                row += [len(arr)]
            row += [p, sig(p), cohens_d(*test_arrays)]
            out.append(row)
            
    df = pd.DataFrame(out, columns=col_names)
    if adjust:
        df['padj'] = np.NaN * np.ones(len(df))
        for _, grp_idx in df.groupby(column_groups).groups.items():
            padj = fwer_sequential_adjust(df.ix[grp_idx])
            df.loc[padj.index, 'padj'] = padj
        df['sig_adj'] = df['padj'].apply(sig)
   
    idx_cols = ['test']
    if groups:
        idx_cols += list(column_groups)
    df = df.set_index(idx_cols)
    df = df.sortlevel()
    return df

def describe(arr):
    funcs = [np.size, np.min, np.max, np.mean, np.median, np.var,
             scipy.stats.skew, scipy.stats.kurtosis]

    out = [(f.func_name, f(arr)) for f in funcs]
    return pd.DataFrame(out, columns=['statistic', 'value']).set_index('statistic')


def icc_3_1(Y):
    '''
    Calculates Interclass Correlation Coefficient (3,1) as defined in
    P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses
    in Assessing Rater Reliability".
    This particular implementation is aimed at relaibility (test-retest) studies.
    @see https://github.com/nipy/nipype/blob/44f75665bf4776d7e6ff63395254de3111da8ce5/nipype/algorithms/icc.py#L23
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, scipy.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE #variance of error
    r_var = (MSR - MSE)/nb_conditions #variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe
