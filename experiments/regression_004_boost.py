from argparse import ArgumentParser
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from alpaca.utils.datasets.builder import build_dataset
from ngboost import NGBRegressor

from regression_002_mc import manual_seed, split_and_scale, uq_ll


def main(name, repeats):
    manual_seed(42)
    dataset = build_dataset(name, val_split=0)
    x, y = dataset.dataset('train')
    result_errors = []
    result_ll = []
    for i in range(repeats):
        print(i)
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        y_train = y_train[:, 0]

        ngb = NGBRegressor().fit(x_train, y_train)
        Y_dists = ngb.pred_dist(x_test)

        y_test_unscaled = y_scaler.inverse_transform(y_test[:, 0])
        y_preds = y_scaler.inverse_transform(ngb.predict(x_test))

        rmse = np.sqrt(mean_squared_error(y_preds, y_test_unscaled))
        errors = np.abs(y_preds - y_test_unscaled)
        uncertainty = Y_dists.scale * y_scaler.scale_

        result_errors.append(rmse)
        result_ll.append(uq_ll(errors, uncertainty))

    print()
    print(name)
    print(np.mean(result_errors), np.std(result_errors))
    print(np.mean(result_ll), np.std(result_ll))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()

    main(args.name, args.repeats)
