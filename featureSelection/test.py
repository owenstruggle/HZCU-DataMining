import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_data(dataset):
    new_data = pd.DataFrame()
    for one in dataset.columns:
        col = dataset[one]
        if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or \
                (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
            new_data[one] = dataset[one]
        else:
            keys = list(set(list(col)))
            values = list(range(len(keys)))
            new = dict(zip(keys, values))
            new_data[one] = dataset[one].map(new)
    return new_data


if __name__ == '__main__':
    data = pd.read_csv('data/german_clean.csv').iloc[:, :20]
    target = pd.read_csv('data/german_clean.csv').iloc[:, 20:].to_numpy()

    feature_list = ['credit_amount_new', 'savings_status', 'job', 'foreign_worker', 'purpose', 'existing_credits',
                    'checking_status', 'duration_new', 'age', 'credit_amount']
    data = get_data(data).to_numpy()

    target[target == 1] = 0
    target[target == 2] = 1

    train_data = data[:700, :]
    train_target = target[:700, :]
    test_data = data[700:, :]
    test_target = target[700:, :]

    reg = LogisticRegression(max_iter=1000)
    reg.fit(train_data, train_target)
    print(reg.score(test_data, test_target))
