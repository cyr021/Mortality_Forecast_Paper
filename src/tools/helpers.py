def know_more_data(df):
    print('df.columns:\n', df.columns, '\n',
          'statistics summary:\n', df['language'].describe())
    # print([df[i].describe() for i in list(df.columns)])
    for i in list(df.columns):
        print(i)
        print(df[i].unique())
        print('\n')