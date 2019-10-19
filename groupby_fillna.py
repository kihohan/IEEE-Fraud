df = pd.DataFrame({'grp': ['a', 'a', 'a', 'a'],
                   'col_1': np.random.randn(4)})

df.loc[[1, 3], ['col_1']] = np.nan

 df.groupby('grp').mean()

fill_mean_func = lambda g: g.fillna(g.mean())
df.groupby('grp').apply(fill_mean_func)
