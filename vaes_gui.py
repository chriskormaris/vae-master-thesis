import tkinter as tk
import webbrowser
from tkinter import ttk

import src.knn_missing_values.binarized_mnist
import src.knn_missing_values.cifar10
import src.knn_missing_values.mnist
import src.knn_missing_values.movielens
import src.knn_missing_values.omniglot
import src.knn_missing_values.orl_faces
import src.knn_missing_values.yale_faces
import src.vaes_in_keras.binarized_mnist
import src.vaes_in_keras.cifar10
import src.vaes_in_keras.mnist
import src.vaes_in_keras.omniglot
import src.vaes_in_keras.orl_faces
import src.vaes_in_keras.yale_faces
import src.vaes_in_pytorch.binarized_mnist
import src.vaes_in_pytorch.cifar10
import src.vaes_in_pytorch.mnist
import src.vaes_in_pytorch.omniglot
import src.vaes_in_pytorch.orl_faces
import src.vaes_in_pytorch.yale_faces
import src.vaes_in_tensorflow.binarized_mnist
import src.vaes_in_tensorflow.cifar10
import src.vaes_in_tensorflow.mnist
import src.vaes_in_tensorflow.omniglot
import src.vaes_in_tensorflow.orl_faces
import src.vaes_in_tensorflow.yale_faces
import src.vaes_missing_values_in_pytorch.binarized_mnist
import src.vaes_missing_values_in_pytorch.cifar10
import src.vaes_missing_values_in_pytorch.mnist
import src.vaes_missing_values_in_pytorch.movielens
import src.vaes_missing_values_in_pytorch.omniglot
import src.vaes_missing_values_in_pytorch.orl_faces
import src.vaes_missing_values_in_pytorch.yale_faces
import src.vaes_missing_values_in_tensorflow.binarized_mnist
import src.vaes_missing_values_in_tensorflow.cifar10
import src.vaes_missing_values_in_tensorflow.mnist
import src.vaes_missing_values_in_tensorflow.movielens
import src.vaes_missing_values_in_tensorflow.omniglot
import src.vaes_missing_values_in_tensorflow.orl_faces
import src.vaes_missing_values_in_tensorflow.yale_faces
from src.utilities.constants import *

# global variables #
isAlgorithmSelected = False
isDatasetSelected = False


#####

# functions #

def run(
        algorithm,
        dataset,
        latent_dim,
        epochs,
        learning_rate,
        batch_size,
        K,
        mnist_digits_or_fashion,
        cifar_rgb_or_grayscale,
        omniglot_language,
        missing_values_construction
):
    if 'knn' not in algorithm.lower():
        arguments = [latent_dim, epochs, batch_size]
        if 'keras' not in algorithm.lower():
            arguments.extend([learning_rate])
        if 'missing' in algorithm.lower() and dataset is not None and 'movielens' not in dataset.lower():
            arguments.extend([missing_values_construction])
        if dataset_var.get() == 'mnist':
            arguments.extend([mnist_digits_or_fashion])
        elif dataset_var.get() == 'cifar10':
            arguments.extend([cifar_rgb_or_grayscale])
        elif dataset_var.get() == 'omniglot':
            arguments.extend([omniglot_language])
    else:
        arguments = [K]
        if 'missing' in algorithm.lower() and dataset is not None and 'movielens' not in dataset.lower():
            arguments.extend([missing_values_construction])
        if dataset_var.get() == 'mnist':
            arguments.extend([mnist_digits_or_fashion])
        elif dataset_var.get() == 'cifar10':
            arguments.extend([cifar_rgb_or_grayscale])
        elif dataset_var.get() == 'omniglot':
            arguments.extend([omniglot_language])

    print('********************')
    print(f'Running {algorithm}.{dataset}')
    print(f'arguments: {arguments}')

    if algorithm == 'knn_missing_values':
        if dataset == 'binarized_mnist':
            src.knn_missing_values.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.knn_missing_values.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.knn_missing_values.mnist.mnist(*arguments)
        elif dataset == 'movielens':
            src.knn_missing_values.movielens.movielens(*arguments)
        elif dataset == 'omniglot':
            src.knn_missing_values.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.knn_missing_values.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.knn_missing_values.yale_faces.yale_faces(*arguments)
    elif algorithm == 'vaes_in_keras':
        if dataset == 'binarized_mnist':
            src.vaes_in_keras.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.vaes_in_keras.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.vaes_in_keras.mnist.mnist(*arguments)
        elif dataset == 'omniglot':
            src.vaes_in_keras.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.vaes_in_keras.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.vaes_in_keras.yale_faces.yale_faces(*arguments)
    elif algorithm == 'vaes_in_pytorch':
        if dataset == 'binarized_mnist':
            src.vaes_in_pytorch.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.vaes_in_pytorch.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.vaes_in_pytorch.mnist.mnist(*arguments)
        elif dataset == 'omniglot':
            src.vaes_in_pytorch.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.vaes_in_pytorch.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.vaes_in_pytorch.yale_faces.yale_faces(*arguments)
    elif algorithm == 'vaes_in_tensorflow':
        if dataset == 'binarized_mnist':
            src.vaes_in_tensorflow.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.vaes_in_tensorflow.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.vaes_in_tensorflow.mnist.mnist(*arguments)
        elif dataset == 'omniglot':
            src.vaes_in_tensorflow.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.vaes_in_tensorflow.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.vaes_in_tensorflow.yale_faces.yale_faces(*arguments)
    elif algorithm == 'vaes_missing_values_in_pytorch':
        if dataset == 'binarized_mnist':
            src.vaes_missing_values_in_pytorch.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.vaes_missing_values_in_pytorch.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.vaes_missing_values_in_pytorch.mnist.mnist(*arguments)
        elif dataset == 'movielens':
            src.vaes_missing_values_in_pytorch.movielens.movielens(*arguments)
        elif dataset == 'omniglot':
            src.vaes_missing_values_in_pytorch.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.vaes_missing_values_in_pytorch.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.vaes_missing_values_in_pytorch.yale_faces.yale_faces(*arguments)
    elif algorithm == 'vaes_missing_values_in_tensorflow':
        if dataset == 'binarized_mnist':
            src.vaes_missing_values_in_tensorflow.binarized_mnist.binarized_mnist(*arguments)
        elif dataset == 'cifar10':
            src.vaes_missing_values_in_tensorflow.cifar10.cifar10(*arguments)
        elif dataset == 'mnist':
            src.vaes_missing_values_in_tensorflow.mnist.mnist(*arguments)
        elif dataset == 'movielens':
            src.vaes_missing_values_in_tensorflow.movielens.movielens(*arguments)
        elif dataset == 'omniglot':
            src.vaes_missing_values_in_tensorflow.omniglot.omniglot(*arguments)
        elif dataset == 'orl_faces':
            src.vaes_missing_values_in_tensorflow.orl_faces.orl_faces(*arguments)
        elif dataset == 'yale_faces':
            src.vaes_missing_values_in_tensorflow.yale_faces.yale_faces(*arguments)


def hide_extra_options():
    mnistDatasetFrame.pack_forget()
    cifarDatasetFrame.pack_forget()
    omniglotDatasetFrame.pack_forget()
    missingValuesFrame.pack_forget()


def check_algorithm_and_show_vae_frame():
    welcomeFrame.pack_forget()

    hide_extra_options()

    kNNFrame.pack_forget()
    runFrame.pack_forget()
    vaeFrame.pack()
    global isAlgorithmSelected
    isAlgorithmSelected = True
    if 'keras' in algorithm_var.get().lower():
        vae_empty_line_label.pack_forget()
        learning_rate_label.pack_forget()
        learning_rate_frame.pack_forget()
        learning_rate_text.pack_forget()
    else:
        vae_empty_line_label.pack_forget()
        learning_rate_label.pack()
        learning_rate_frame.pack()
        learning_rate_text.pack()
        vae_empty_line_label.pack()
    if 'missing' in algorithm_var.get().lower():
        datasetsMenu.entryconfig(6, state='normal')  # enable 'MovieLens' dataset
    else:
        datasetsMenu.entryconfig(6, state='disabled')  # disable 'MovieLens' dataset
    if 'missing' in algorithm_var.get().lower() and \
            not dataset_var.get() == 'movielens' and not dataset_var.get() == '':
        missingValuesFrame.pack()
    if dataset_var.get() == 'mnist':
        mnistDatasetFrame.pack()
    if dataset_var.get() == 'cifar10':
        cifarDatasetFrame.pack()
    elif dataset_var.get() == 'omniglot':
        omniglotDatasetFrame.pack()
    if isAlgorithmSelected and isDatasetSelected:
        runFrame.pack(side='bottom')
        status.config(text='Algorithm selected: ' + algorithm_var.get() + ', dataset selected: ' + dataset_var.get())


def check_algorithm_and_show_knn_frame():
    welcomeFrame.pack_forget()

    hide_extra_options()

    vaeFrame.pack_forget()
    runFrame.pack_forget()
    kNNFrame.pack()
    global isAlgorithmSelected
    isAlgorithmSelected = True
    if 'missing' in algorithm_var.get().lower():
        datasetsMenu.entryconfig(6, state='normal')  # enable 'MovieLens' dataset
    else:
        datasetsMenu.entryconfig(6, state='disabled')  # disable 'MovieLens' dataset
    if 'missing' in algorithm_var.get().lower() and \
            not dataset_var.get() == 'movielens' and not dataset_var.get() == '':
        missingValuesFrame.pack()
    if dataset_var.get() == 'mnist':
        mnistDatasetFrame.pack()
    if dataset_var.get() == 'cifar10':
        cifarDatasetFrame.pack()
    elif dataset_var.get() == 'omniglot':
        omniglotDatasetFrame.pack()
    if isAlgorithmSelected and isDatasetSelected:
        runFrame.pack(side='bottom')
        status.config(text='Algorithm selected: ' + algorithm_var.get() + ', dataset selected: ' + dataset_var.get())


def check_dataset():
    global isDatasetSelected

    hide_extra_options()

    isDatasetSelected = True

    if 'missing' in algorithm_var.get().lower() and \
            not dataset_var.get() == 'movielens' and not dataset_var.get() == '':
        missingValuesFrame.pack()
    if dataset_var.get() == 'mnist' and not welcomeFrame.winfo_ismapped():
        mnistDatasetFrame.pack()
    elif dataset_var.get() == 'cifar10' and not welcomeFrame.winfo_ismapped():
        cifarDatasetFrame.pack()
    elif dataset_var.get() == 'omniglot' and not welcomeFrame.winfo_ismapped():
        omniglotDatasetFrame.pack()
    if isAlgorithmSelected and isDatasetSelected:
        runFrame.pack(side='bottom')
        status.config(text='Algorithm selected: ' + algorithm_var.get() + ', dataset selected: ' + dataset_var.get())


# center the window on screen
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def about_window():
    window = tk.Toplevel(root)

    # change title
    window.wm_title('About')

    window.resizable(False, False)

    creator = tk.Label(window, text=f'© Created by: {author}')
    creator.pack()

    professor = tk.Label(window, text='Supervisor Professor: Dr. Michalis Titsias')
    professor.pack()

    thesis = tk.Label(window, text='Thesis on Variational Autoencoders & Missing Values Completion Algorithms')
    thesis.pack()

    university = tk.Label(window, text='Athens University of Economics & Business')
    university.pack()

    msc = tk.Label(window, text='MSc in Computer Science')
    msc.pack()

    date = tk.Label(window, text='Date: April 2018')
    date.pack()

    version_label = tk.Label(window, text=f'Version: {version}')
    version_label.pack()

    # change icon
    window.iconbitmap(icons_path + 'info.ico')

    okButton = tk.Button(
        window,
        text='Ok',
        fg='#340DFD',
        bg='#5BFFAC',
        height=2,
        width=6,
        command=window.destroy
    )
    okButton.pack(side=tk.BOTTOM)

    # make the child window transient to the root
    window.transient(root)
    window.grab_set()
    center(window)
    root.wait_window(window)


def datasets_details_window():
    window = tk.Toplevel(root)

    # change title
    window.wm_title('Datasets Details')

    window.resizable(False, False)

    mnist_ds_label = tk.Label(
        window,
        text='MNIST dataset\n'
             '# TRAIN data: 55000, # TEST data: 10000\n'
             '# VALIDATION data: 5000 # Classes: 10\n'
             'Dimensions: 28x28 pixels'
    )
    mnist_ds_label.pack()
    mnist_ds_link = tk.Label(window, text='MNIST dataset link', fg='blue', cursor='hand2')
    mnist_ds_link.pack()
    mnist_ds_link.bind('<Button-1>', mnist_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    fashion_mnist_ds_label = tk.Label(
        window,
        text='Fashion MNIST dataset\n'
             '# TRAIN data: 60000, # TEST data: 10000\n'
             '# Classes: 10, Dimensions: 28x28 pixels'
    )
    fashion_mnist_ds_label.pack()
    fashion_mnist_ds_link = tk.Label(window, text='Fashion MNIST dataset link', fg='blue', cursor='hand2')
    fashion_mnist_ds_link.pack()
    fashion_mnist_ds_link.bind('<Button-1>', fashion_mnist_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    binarized_mnist_ds_label = tk.Label(
        window,
        text='Binarized MNIST dataset\n'
             '# TRAIN data: 50000, # TEST data: 10000\n'
             '# VALIDATION data: 10000, # Classes: 10\n'
             'Dimensions: 28x28 pixels'
    )
    binarized_mnist_ds_label.pack()
    binarized_mnist_ds_link = tk.Label(window, text='Binarized MNIST dataset link', fg='blue', cursor='hand2')
    binarized_mnist_ds_link.pack()
    binarized_mnist_ds_link.bind('<Button-1>', binarized_mnist_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    cifar_10_ds_label = tk.Label(
        window,
        text='CIFAR-10 dataset\n'
             '# TRAIN data: 50000, # TEST data: 10000\n'
             '# Classes: 10\n'
             'RGB Dimensions: 32x32x3 pixels\n'
             'Grayscale Dimensions: 32x32x1 pixels'
    )
    cifar_10_ds_label.pack()
    cifar_10_link = tk.Label(window, text='CIFAR-10 dataset link', fg='blue', cursor='hand2')
    cifar_10_link.pack()
    cifar_10_link.bind('<Button-1>', cifar_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    omniglot_ds_label = tk.Label(
        window,
        text='OMNIGLOT dataset\n'
             'English Alphabet\n'
             '# TRAIN data: 390, # TEST data: 130, # Classes: 26\n'
             'Greek Alphabet\n'
             '# TRAIN data: 360, # TEST data: 120, # Classes: 24\n'
             'Dimensions: 28x28 pixels'
    )
    omniglot_ds_label.pack()
    omniglot_ds_link = tk.Label(window, text='OMNIGLOT dataset link', fg='blue', cursor='hand2')
    omniglot_ds_link.pack()
    omniglot_ds_link.bind('<Button-1>', omniglot_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    yale_ds_label = tk.Label(
        window,
        text='YALE Faces dataset\n'
             '# of data: 2442, # Classes: 38\n'
             'Dimensions: 168x192 pixels'
    )
    yale_ds_label.pack()
    yale_ds_link = tk.Label(window, text='YALE Faces dataset link', fg='blue', cursor='hand2')
    yale_ds_link.pack()
    yale_ds_link.bind('<Button-1>', yale_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    the_db_of_faces_ds_label = tk.Label(
        window,
        text='ORL Face Database\n'
             '# of data: 400, # Classes: 40\n'
             'Dimensions: 92x112 pixels'
    )
    the_db_of_faces_ds_label.pack()
    the_db_of_faces_ds_link = tk.Label(window, text='ORL Face Database link', fg='blue', cursor='hand2')
    the_db_of_faces_ds_link.pack()
    the_db_of_faces_ds_link.bind('<Button-1>', the_db_of_faces_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    movielens_ds_label = tk.Label(
        window,
        text='MovieLens 100k dataset\n'
             '# TRAIN ratings: 90570, # TEST ratings: 9430\n'
             '# of users: 943, # of movies: 1682\n'
             '# of total ratings: 1586126, non-missing percentage: 5.7 %'
    )
    movielens_ds_label.pack()
    movielens_ds_link = tk.Label(window, text='MovieLens dataset link', fg='blue', cursor='hand2')
    movielens_ds_link.pack()
    movielens_ds_link.bind('<Button-1>', movielens_link_command)
    sep = ttk.Separator(window, orient='horizontal')
    sep.pack(fill='x')

    download_all_datasets_link = tk.Label(window, text='Download all datasets here', fg='blue', cursor='hand2')
    download_all_datasets_link.pack()
    download_all_datasets_link.bind('<Button-1>', download_all_datasets_command)

    # change icon
    window.iconbitmap(icons_path + 'help.ico')

    # make the child window transient to the root
    window.transient(root)
    window.grab_set()
    center(window)
    root.wait_window(window)


def mnist_link_command(event):
    webbrowser.open_new(r'http://yann.lecun.com/exdb/mnist')


def fashion_mnist_link_command(event):
    webbrowser.open_new(r'https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion')


def binarized_mnist_link_command(event):
    webbrowser.open_new(r'https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST')


def cifar_link_command(event):
    webbrowser.open_new(r'https://www.cs.toronto.edu/~kriz/cifar.html')


def omniglot_link_command(event):
    webbrowser.open_new(r'https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT')


def yale_link_command(event):
    webbrowser.open_new(r'https://vision.ucsd.edu/content/extended-yale-face-database-b-b')


def the_db_of_faces_link_command(event):
    webbrowser.open_new(r'http://cam-orl.co.uk/facedatabase.html')


def movielens_link_command(event):
    webbrowser.open_new(r'https://grouplens.org/datasets/movielens')


def download_all_datasets_command(event):
    webbrowser.open_new(r'https://www.dropbox.com/sh/ucvad0dkcbxuyho/AAAjjrRPYiGLLPc_VKru4-Uva?dl=0')


if __name__ == '__main__':
    # create window and set title
    root = tk.Tk()
    root.title('vaes')

    # change window size
    root.geometry('800x810')
    center(root)

    # change icon
    root.iconbitmap(icons_path + 'vaes.ico')

    # Frames #
    welcomeFrame = tk.Frame(root)
    vaeFrame = tk.Frame(root)
    kNNFrame = tk.Frame(root)

    mnistDatasetFrame = tk.Frame(root)
    cifarDatasetFrame = tk.Frame(root)
    omniglotDatasetFrame = tk.Frame(root)
    missingValuesFrame = tk.Frame(root)

    runFrame = tk.Frame(root)

    # Widgets #

    # 1. welcomeFrame Widgets #
    empty_line_label = tk.Label(welcomeFrame, text='\n')
    empty_line_label.pack()

    aueb_logo = tk.PhotoImage(file=icons_path + 'aueb_logo.png')
    image_label = tk.Label(welcomeFrame, image=aueb_logo, anchor=tk.CENTER)
    image_label.pack()

    welcome_label = tk.Label(welcomeFrame, text='Welcome to the Variational Autoencoders graphical user interface.')
    welcome_label.pack()
    instructions_label = tk.Label(
        welcomeFrame,
        text='Please select an algorithm and a dataset from the dropdown menus at the top.'
    )
    instructions_label.pack()

    # show welcomeFrame
    welcomeFrame.pack()

    # 2. vaeFrame Widgets #
    # tkinter variables
    algorithm_var = tk.StringVar(root)
    dataset_var = tk.StringVar(root)

    latent_dim_var = tk.IntVar(root, 64)
    epochs_var = tk.IntVar(root, 100)
    learning_rate_var = tk.DoubleVar(root, 0.01)
    batch_size_var = tk.StringVar(root, '250')
    K_var = tk.IntVar(root, 10)

    mnist_digits_or_fashion_var = tk.StringVar(root, 'digits')
    cifar_rgb_or_grayscale_var = tk.StringVar(root, 'grayscale')
    omniglot_language_var = tk.StringVar(root, 'english')
    missing_values_construction_var = tk.StringVar(root, 'structured')

    latent_dim_label = tk.Label(vaeFrame, text='latent dimension:')
    latent_dim_label.pack()
    for i in [32, 64, 128]:
        tk.Radiobutton(
            vaeFrame,
            text=i,
            padx=2,
            variable=latent_dim_var,
            value=i
        ).pack(anchor=tk.CENTER)
    latent_dim_text = tk.Entry(vaeFrame, textvariable=latent_dim_var)
    latent_dim_text.pack()

    vae_empty_line_label = tk.Label(vaeFrame, text='\r')
    vae_empty_line_label.pack()

    epochs_label = tk.Label(vaeFrame, text='epochs:')
    epochs_label.pack()
    for i in [20, 50, 100, 200]:
        tk.Radiobutton(
            vaeFrame,
            text=i,
            padx=2,
            variable=epochs_var,
            value=i
        ).pack(anchor=tk.CENTER)
    epochs_text = tk.Entry(vaeFrame, textvariable=epochs_var)
    epochs_text.pack()

    vae_empty_line_label = tk.Label(vaeFrame, text='\r')
    vae_empty_line_label.pack()

    batch_size_label = tk.Label(vaeFrame, text='batch size:')
    batch_size_label.pack()
    for value in [250, 500, 'N']:
        tk.Radiobutton(
            vaeFrame,
            text=value,
            padx=2,
            variable=batch_size_var,
            value=value
        ).pack(anchor=tk.CENTER)
    batch_size_text = tk.Entry(vaeFrame, textvariable=batch_size_var)
    batch_size_text.pack()

    vae_empty_line_label = tk.Label(vaeFrame, text='\r')
    vae_empty_line_label.pack()

    learning_rate_label = tk.Label(vaeFrame, text='learning rate:')
    learning_rate_label.pack()
    learning_rate_frame = tk.Frame(vaeFrame)
    learning_rate_frame.pack()
    for value in [0.1, 0.01, 0.001]:
        tk.Radiobutton(
            learning_rate_frame,
            text=value,
            padx=2,
            variable=learning_rate_var,
            value=value
        ).pack(anchor=tk.CENTER)
    learning_rate_text = tk.Entry(vaeFrame, textvariable=learning_rate_var)
    learning_rate_text.pack()

    # 3. kNNFrame Widgets #
    k_label = tk.Label(kNNFrame, text='K:')
    k_label.pack()
    for value in [1, 3, 10, 100]:
        tk.Radiobutton(
            kNNFrame,
            text=value,
            padx=2,
            variable=K_var,
            value=value
        ).pack(anchor=tk.CENTER)
    k_text = tk.Entry(kNNFrame, textvariable=K_var)
    k_text.pack()

    vae_empty_line_label = tk.Label(vaeFrame, text='\r')
    vae_empty_line_label.pack()

    knn_empty_line_label = tk.Label(kNNFrame, text='\r')
    knn_empty_line_label.pack()

    # 4. mnistDatasetFrame Widgets #
    mnist_label = tk.Label(mnistDatasetFrame, text='Digits or Fashion:')
    mnist_label.pack()
    for value in ['digits', 'fashion']:
        tk.Radiobutton(
            mnistDatasetFrame,
            text=value,
            padx=2,
            variable=mnist_digits_or_fashion_var,
            value=value.lower()
        ).pack(anchor=tk.CENTER)

    # 5. cifarDatasetFrame Widgets #
    cifar_label = tk.Label(cifarDatasetFrame, text='Grayscale or RGB:')
    cifar_label.pack()
    for value in ['grayscale', 'RGB']:
        tk.Radiobutton(
            cifarDatasetFrame,
            text=value,
            padx=2,
            variable=cifar_rgb_or_grayscale_var,
            value=value.lower()
        ).pack(anchor=tk.CENTER)

    # 6. omniglotDatasetFrame Widgets #
    omniglot_label = tk.Label(omniglotDatasetFrame, text='Language:')
    omniglot_label.pack()
    for value in ['English', 'Greek']:
        tk.Radiobutton(
            omniglotDatasetFrame,
            text=value,
            padx=2,
            variable=omniglot_language_var,
            value=value.lower()
        ).pack(anchor=tk.CENTER)

    # 7. missing values Widgets #
    missing_values_label = tk.Label(missingValuesFrame, text='missing values construction:')
    missing_values_label.pack()
    for value in ['structured', 'random']:
        tk.Radiobutton(
            missingValuesFrame,
            text=value,
            padx=2,
            variable=missing_values_construction_var,
            value=value.lower()
        ).pack(anchor=tk.CENTER)
    empty_line_label = tk.Label(missingValuesFrame, text='\r')
    empty_line_label.pack()

    # Status Bar #

    status = tk.Label(runFrame, bd=1, relief=tk.SUNKEN, anchor=tk.S)
    status.pack(side=tk.BOTTOM, fill=tk.X)

    # Menus #

    menu = tk.Menu(root)
    root.config(menu=menu)

    algorithms = {
        'vaes_in_tensorflow': 'VAE in TensorFlow',
        'vaes_in_pytorch': 'VAE in PyTorch',
        'vaes_in_keras': 'VAE in Keras',
        'vaes_missing_values_in_tensorflow': 'VAE Missing Values completion algorithm in TensorFlow',
        'vaes_missing_values_in_pytorch': 'VAE Missing Values completion algorithm in PyTorch',
        'knn_missing_values': 'K-NN Missing Values completion algorithm'
    }

    algorithmsMenu = tk.Menu(menu, tearoff=False)
    menu.add_cascade(label='Algorithms', menu=algorithmsMenu)  # adds drop-down menu
    for name in algorithms:
        description = algorithms[name]
        if name == 'vaes_missing_values_in_tensorflow':
            algorithmsMenu.add_separator()
        if 'knn' in name.lower():
            algorithmsMenu.add_radiobutton(
                label=description,
                variable=algorithm_var,
                value=name,
                command=check_algorithm_and_show_knn_frame
            )
        else:
            algorithmsMenu.add_radiobutton(
                label=description,
                variable=algorithm_var,
                value=name,
                command=check_algorithm_and_show_vae_frame
            )

    datasets = {
        'mnist': 'MNIST',
        'binarized_mnist': 'Binarized MNIST',
        'cifar10': 'CIFAR-10',
        'omniglot': 'OMNIGLOT',
        'yale_faces': 'YALE Faces',
        'orl_faces': 'ORL Face Database',
        'movielens': 'MovieLens'
    }

    datasetsMenu = tk.Menu(menu, tearoff=False)
    menu.add_cascade(label='Datasets', menu=datasetsMenu)  # adds drop-down menu
    for name in datasets:
        description = datasets[name]
        if name != 'movielens':
            datasetsMenu.add_radiobutton(
                label=description,
                variable=dataset_var,
                value=name,
                command=check_dataset
            )
        else:
            # Leave 'MovieLens' dataset disabled initially.
            datasetsMenu.add_radiobutton(
                label=description,
                variable=dataset_var,
                value=name,
                command=check_dataset,
                state='disabled'
            )

    aboutMenu = tk.Menu(menu, tearoff=False)
    menu.add_cascade(label='About', menu=aboutMenu)  # adds drop-down menu
    aboutMenu.add_command(label='About', command=about_window)
    aboutMenu.add_command(label='Datasets Details', command=datasets_details_window)
    aboutMenu.add_command(label='Exit', command=root.quit)

    runButton = tk.Button(
        runFrame,
        text='Run',
        fg='#340DFD',
        bg='#5BFFAC',
        height=2,
        width=6,
        command=lambda: run(
            algorithm_var.get(),
            dataset_var.get(),
            latent_dim_var.get(),
            epochs_var.get(),
            learning_rate_var.get(),
            batch_size_var.get(),
            K_var.get(),
            mnist_digits_or_fashion_var.get(),
            cifar_rgb_or_grayscale_var.get(),
            omniglot_language_var.get(),
            missing_values_construction_var.get()
        )
    )
    runButton.pack(side=tk.BOTTOM)

    center(root)
    root.mainloop()
