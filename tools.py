import matplotlib.pyplot as plt
import numpy as np
from load_data import load_dataset

def show_results(num_img, x, y_true, y_pred):
    """
    num_img: количество изображений, произвольно набранных из x
    x: набор изображений (4х мерная матрица)
    """

    imgs = []
    some_y_true = []
    for _ in range(num_img):
        index = np.random.randint(0, x.shape[0])
        imgs.append(x[index])
        some_y_true.append(y_true[index])

    imgs = np.array(imgs)
    some_y_true = np.array(some_y_true)

    fig, axes = plt.subplots(2,num_img)
    fig.set_size_inches(15,10)
    for i,axes_rows in enumerate(axes):
        for j,ax in enumerate(axes_rows):
            ax.imshow(imgs[j][:,:,0], cmap='gray')
            if i == 0:
                y = some_y_true
            else:
                y = y_pred

            ax.scatter([y[j][k]*256 for k in range(0,8,2)], [y[j][k]*256 for k in range(1,8,2)])

            ax.axes.xaxis.set_visible(False)
            ax.get_yaxis().set_ticks([])
    axes[0][0].set_ylabel('Истинная разметка', size=15, weight='bold', color='green')
    axes[-1][0].set_ylabel('Предсказанная разметка', size=15, weight='bold', color='red')