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
    print()
    print(clf)
    for mag, folds in r.items():
        print(mag, end=": ")
        avg_accuracy = 0
        for fold, data in folds.items():
            avg_accuracy += data["accuracy"]

        if len(d) > 0 and len(folds) > 0:
            avg_accuracy = avg_accuracy / len(folds)
            print("avg acc:", avg_accuracy)
