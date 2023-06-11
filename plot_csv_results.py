import sys
import csv
import matplotlib.pyplot as plt
import os

colors = ('r', 'g', 'b', 'c', 'm')
lines = ('solid', 'dashed', 'dotted', 'dashdot')

def plot_single_plots(runs, path_to_save): 
    c = colors[2]
    for run in runs:
        x = run['epoch']
        y_train = run['train']
        y_val = run['val']

        plt.plot(x, y_train, label="Trening", color=c, linestyle='dashed')
        plt.plot(x, y_val, label="Ewaluacja", color=c, linestyle='solid')
        # plt.xlim([1, max(x)])
        plt.ylim([min(y_train + y_val) - 0.03, max(y_train + y_val) + 0.03])
        xticks = range(1, max(x) + 1) if max(x) < 20 else range(1, max(x) + 1, 2)
        plt.xticks(xticks)
        plt.legend()
        # plt.show()
        plt.savefig(f'saved_imgs/{run["path"]}.png')
        with open(f'saved_imgs/{run["path"]}.txt', 'w') as f:
            head = "epoch, train, eval"
            start = f"1, {run['train'][0]}, {run['val'][0]}"
            best_val_loss = min(run['val'])
            best_val_loss_epoch = run['val'].index(best_val_loss) + 1
            best = f"{best_val_loss_epoch}, {run['train'][best_val_loss_epoch - 1]}, {best_val_loss}"
            end = f"{run['epoch'][-1]}, {run['train'][-1]}, {run['val'][-1]}"
            print(f"{head}\n{start}\n{best}\n{end}", file=f)
        plt.clf()


def plot_results(cnn, gnn):
    c_gnn = colors[2]
    c_cnn = colors[1]
    x_cnn = cnn['epoch']
    y_train_cnn = cnn['train']
    y_val_cnn = cnn['val']

    x_gnn = gnn['epoch']
    y_train_gnn = gnn['train']
    y_val_gnn = gnn['val']

    plt.plot(x_gnn, y_train_gnn, label="Trening GNN", color=c_gnn, linestyle='dashed')
    plt.plot(x_gnn, y_val_gnn, label="Ewaluacja GNN", color=c_gnn, linestyle='solid')

    plt.plot(x_cnn, y_train_cnn, label="Trening CNN", color=c_cnn, linestyle='dashed')
    plt.plot(x_cnn, y_val_cnn, label="Ewaluacja CNN", color=c_cnn, linestyle='solid')

    plt.legend()
    # plt.show()
    plt.savefig(f'saved_imgs/gnn_cnn_compare_plot.png')
    plt.clf()


def read_files(paths):
    runs = []
    for path in paths:
        run = {'epoch': [], 'total': [], 'train': [], 'val': [], 'path': os.path.splitext(os.path.basename(path))[0]}
        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, line in enumerate(reader):
                if len(line) == 4:
                    assert line[0] == str(i + 1)
                    run['epoch'].append(int(line[0]))
                    run['total'].append(int(line[1]))
                    run['train'].append(round(float(line[2]), 4))
                    run['val'].append(round(float(line[3]), 4))
                    
        runs.append(run)
    return runs

if __name__ == '__main__':
    csv_to_plot = []
    for path in sys.argv[1:]:
        csv_to_plot.append(path)
    runs = read_files(csv_to_plot)
    # plot_single_plots(runs)
    plot_results(runs[0], runs[1])
    