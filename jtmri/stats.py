import numpy as np
import pandas as pd
from collections import namedtuple
import logging

log = logging.getLogger('jtmri.stats')


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
   

def tests(data, groups, tests, stat, adj):
    '''Generate a table of test statistics.
    Args:
     data   -- Pandas dataframe
     groups -- Groups
     tests  -- Tests object
     stat   -- Statistics to perform for each test
     adj    -- Adjust p-values if True
    '''
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
            row += list(key)
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
