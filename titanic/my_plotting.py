import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

def plot_precision_recall(recall, precision, show=True, filename=None):
    step_kwargs = {'step': 'post'}

    plt.step(
        recall,
        precision,
        color='b',
        alpha=0.2,
        where='post'
    )

    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if filename:
        plt.savefig(filename)

    if show:
        plt.show()
    plt.close()

def plot_precision_recall_from_model(trained_model, train_set, targets, show=True, filename=None):
    probas = trained_model.predict_proba(train_set)
    precision, recall, thresholds = precision_recall_curve(
        targets,
        probas[:, 1],
    )
    plot_precision_recall(recall, precision, show, filename)


