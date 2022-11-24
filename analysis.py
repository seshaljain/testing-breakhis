import os
import pickle


def combine_into(d: dict, combined: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            combine_into(v, combined.setdefault(k, {}))
        else:
            combined[k] = v


results = {}
results_files = os.listdir('results')

for result in results_files:
    d = pickle.load(open(f'results/{result}', 'rb'))
    combine_into(d, results)

for clf, r in results.items():
    for mag, folds in r.items():
        avg_accuracy = 0
        for fold, data in folds.items():
            avg_accuracy += data["accuracy"]

        if (len(d) > 0):
            avg_accuracy = avg_accuracy / len(folds)
            print(clf, mag, "avg accuracy:", avg_accuracy)
