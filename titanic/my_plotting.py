import matplotlib.pyplot as plt

def plot_precision_recall(recall, precision):
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
    plt.show()

