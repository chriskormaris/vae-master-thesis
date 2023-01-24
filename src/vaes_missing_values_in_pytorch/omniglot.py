import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.utilities.constants import *
from src.utilities.get_omniglot_dataset import get_omniglot_dataset
from src.utilities.plot_utils import plot_images
from src.utilities.utils import construct_missing_data, get_non_zero_percentage, rmse, mae
from src.utilities.vae_in_pytorch import initialize_weights, train


def omniglot(
        latent_dim=64,
        epochs=100,
        batch_size='N',
        learning_rate=0.01,
        structured_or_random='structured',
        language='English'
):
    missing_value = 0.5

    if language.lower() == 'greek':
        output_images_path = output_img_base_path + 'vaes_missing_values_in_pytorch/omniglot_greek'
        alphabet = 20
    else:
        output_images_path = output_img_base_path + 'vaes_missing_values_in_pytorch/omniglot_english'
        alphabet = 31

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # LOAD OMNIGLOT DATASET #
    X_train, y_train = get_omniglot_dataset(
        omniglot_dataset_path + '/chardata.mat',
        train_or_test='train',
        alphabet=alphabet,
        binarize=True
    )
    X_test, y_test = get_omniglot_dataset(
        omniglot_dataset_path + '/chardata.mat',
        train_or_test='test',
        alphabet=alphabet,
        binarize=True
    )

    X_merged = np.concatenate((X_train, X_test), axis=0)
    y_merged = np.concatenate((y_train, y_test), axis=0)

    #####

    X_merged_missing, X_merged, y_merged = construct_missing_data(
        X_merged,
        y_merged,
        structured_or_random=structured_or_random
    )

    #####

    N = X_merged.shape[0]
    input_dim = 784  # D
    # M1: number of neurons in the encoder
    # M2: number of neurons in the decoder
    hidden_encoder_dim = 400  # M1
    hidden_decoder_dim = hidden_encoder_dim  # M2
    # latent_dim = Z_dim
    if batch_size == 'N':
        batch_size = N
    else:
        batch_size = int(batch_size)

    #####

    params, solver = initialize_weights(input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lr=learning_rate)

    start_index = None
    end_index = None
    cur_samples = None
    batch_labels = None
    cur_elbo = None
    masked_batch_data = None

    # X_merged_masked: array with 0s where the pixels are missing
    # and 1s where the pixels are not missing
    X_merged_masked = np.array(X_merged_missing)
    X_merged_masked[np.where(X_merged_masked != missing_value)] = 1
    X_merged_masked[np.where(X_merged_masked == missing_value)] = 0

    non_zero_percentage = get_non_zero_percentage(X_merged_masked)
    print(f'non missing values percentage: {non_zero_percentage} %')

    X_filled = np.array(X_merged_missing)

    print()

    iterations = int(N / batch_size)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for i in range(1, iterations + 1):
            start_index = (i - 1) * batch_size
            end_index = i * batch_size

            batch_data = X_filled[start_index:end_index, :]
            batch_labels = y_merged[start_index:end_index]

            cur_samples, cur_elbo = train(batch_data, batch_size, latent_dim, params, solver)

            masked_batch_data = X_merged_masked[start_index:end_index, :]
            cur_samples = np.multiply(masked_batch_data, batch_data) + np.multiply(1 - masked_batch_data, cur_samples)
            X_filled[start_index:end_index, :] = cur_samples

        print(f'Epoch {epoch} | Loss (ELBO): {cur_elbo}')

        if epoch == 1:
            fig = plot_images(
                X_merged[start_index:end_index, :],
                y_merged[start_index:end_index],
                categories=list(range(1, 11)),
                title='Original Data'
            )
            fig.savefig(f'{output_images_path}/original_data_characters_1-10.png', bbox_inches='tight')
            plt.close()
            fig = plot_images(
                X_merged[start_index:end_index, :],
                y_merged[start_index:end_index],
                categories=list(range(11, 21)),
                title='Original Data'
            )
            fig.savefig(f'{output_images_path}/original_data_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_images(
                    X_merged[start_index:end_index, :],
                    y_merged[start_index:end_index],
                    categories=list(range(21, 24)),
                    title='Original Data'
                )
                fig.savefig(f'{output_images_path}/original_data_characters_21-24.png', bbox_inches='tight')
            else:
                fig = plot_images(
                    X_merged[start_index:end_index, :],
                    y_merged[start_index:end_index],
                    categories=list(range(21, 27)),
                    title='Original Data'
                )
                fig.savefig(f'{output_images_path}/original_data_characters_21-26.png', bbox_inches='tight')
            plt.close()

            fig = plot_images(
                X_merged_missing[start_index:end_index, :],
                y_merged[start_index:end_index],
                categories=list(range(1, 11)),
                title='Original Data'
            )
            fig.savefig(f'{output_images_path}/missing_data_characters_1-10.png', bbox_inches='tight')
            plt.close()
            fig = plot_images(
                X_merged_missing[start_index:end_index, :],
                y_merged[start_index:end_index],
                categories=list(range(11, 21)),
                title='Original Data'
            )
            fig.savefig(f'{output_images_path}/missing_data_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_images(
                    X_merged_missing[start_index:end_index, :],
                    y_merged[start_index:end_index],
                    categories=list(range(21, 25)),
                    title='Original Data'
                )
                fig.savefig(f'{output_images_path}/missing_data_characters_21-24.png', bbox_inches='tight')
            else:
                fig = plot_images(
                    X_merged_missing[start_index:end_index, :],
                    y_merged[start_index:end_index],
                    categories=list(range(21, 27)),
                    title=f'Epoch {str(epoch).zfill(3)}'
                )
                fig.savefig(f'{output_images_path}/missing_data_characters_21-26.png', bbox_inches='tight')
            plt.close()

            fig = plot_images(
                masked_batch_data,
                batch_labels,
                categories=list(range(1, 11)),
                title='Masked Data'
            )
            fig.savefig(
                f'{output_images_path}/masked_data_characters_1-10.png',
                bbox_inches='tight'
            )
            plt.close()
            fig = plot_images(
                masked_batch_data,
                batch_labels,
                categories=list(range(1, 11)),
                title='Masked Data'
            )
            fig.savefig(f'{output_images_path}/masked_data_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_images(
                    masked_batch_data,
                    batch_labels,
                    categories=list(range(1, 11)),
                    title='Masked Data'
                )
                fig.savefig(f'{output_images_path}/masked_data_characters_21-24.png', bbox_inches='tight')
            else:
                fig = plot_images(
                    masked_batch_data,
                    batch_labels,
                    categories=list(range(1, 11)),
                    title='Masked Data'
                )
                fig.savefig(f'{output_images_path}/masked_data_characters_21-26.png', bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == 1:
            fig = plot_images(
                cur_samples,
                batch_labels,
                categories=list(range(1, 11)),
                title=f'Epoch {str(epoch).zfill(3)}'
            )
            fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}_characters_1-10.png', bbox_inches='tight')
            plt.close()
            fig = plot_images(
                cur_samples,
                batch_labels,
                categories=list(range(11, 21)),
                title=f'Epoch {str(epoch).zfill(3)}'
            )
            fig.savefig(f'{output_images_path}/epoch_{str(epoch).zfill(3)}_characters_11-20.png', bbox_inches='tight')
            plt.close()
            if language.lower() == 'greek':
                fig = plot_images(
                    cur_samples,
                    batch_labels,
                    categories=list(range(21, 25)),
                    title=f'Epoch {str(epoch).zfill(3)}'
                )
                fig.savefig(
                    f'{output_images_path}/epoch_{str(epoch).zfill(3)}_characters_21-24.png',
                    bbox_inches='tight'
                )
            else:
                fig = plot_images(
                    cur_samples,
                    batch_labels,
                    categories=list(range(21, 27)),
                    title=f'Epoch {str(epoch).zfill(3)}'
                )
                fig.savefig(
                    f'{output_images_path}/epoch_{str(epoch).zfill(3)}_characters_21-26.png',
                    bbox_inches='tight'
                )
            plt.close()
    elapsed_time = time.time() - start_time

    print(f'training time: {elapsed_time} secs')
    print()

    error1 = rmse(X_merged, X_filled)
    print(f'root mean squared error: {error1}')

    error2 = mae(X_merged, X_filled)
    print(f'mean absolute error: {error2}')


if __name__ == '__main__':
    omniglot()
