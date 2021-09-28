import csv
import numpy as np
import matplotlib.pyplot as plt
import re


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 200):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def parsecol(col):
    pattern = r"\((\d+),\W(-?\d*\.?\d*)\)"
    match = re.search(pattern, col)
    if match:
        x = match.group(1)
        y = match.group(2)
    else:
        x = 0
        y = 0
    return x, y


def main():
    filename = 'data/scores-09-27-2021-15:35:34.csv'
    x_axis = []
    score_history = []
    with open(filename, "r") as thefile:
        reader = csv.reader(thefile)
        for row in reader:
            for col in row:
                (ep, score) = parsecol(col)
                score_history.append(float(score))
                x_axis.append(int(ep))
        plot_learning_curve(x_axis, score_history, "plots/plot1.png")


if __name__ == "__main__":
    main()
