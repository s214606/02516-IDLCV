import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def plot(result_path, test_results, output_path='results/plots/accuracy.png'):
    df = pd.read_csv(result_path)
    df_test = pd.read_csv(test_results)
    plt.title('Accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')
    plt.xlim(1, max(df['epoch']))
    plt.plot(df['epoch'], df['train_accuracy'], label="Train accuracy")
    plt.plot(df_test['epoch'], df_test['test_accuracy'], label="Test accuracy")
    plt.legend()
    plt.savefig(output_path)#, transparent=True)
    plt.clf()

def plot_loss(result_path, test_results, output_path='results/plots/losses.png'):
    df = pd.read_csv(result_path)
    df_test = pd.read_csv(test_results)
    plt.title('Loss plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(1, max(df['epoch']))
    plt.plot(df['epoch'], df['train_avg_loss'], label="Train loss")
    plt.plot(df_test['epoch'], df_test['test_avg_loss'], label="Test loss")
    plt.legend()
    plt.savefig(output_path)#, transparent=True)
    plt.clf()


def visualize_data(dataset, num_samples=5):
    """Visualize a few image–mask pairs from the dataset."""
    # Ensure we don't exceed dataset length
    num_samples = min(num_samples, len(dataset))

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    
    # Handle case when num_samples == 1 (axes not a 2D array)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        image, mask = dataset[i]

        # Convert tensors to numpy arrays
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
            # Move channel dimension to the end if needed (C,H,W → H,W,C)
            if image_np.ndim == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = np.array(image)

        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = np.squeeze(mask_np, 0)
        else:
            mask_np = np.array(mask)

        # Plot image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Plot mask
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title(f"Mask {i}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f'results/{dataset.__class__.__name__}_data_sample.png')



if __name__ == '__main__':

    plot(
        result_path='results/train_results.csv',
        test_results='results/test_results.csv',
        output_path='results/plots/accuracy.png'
        )
    
    plot_loss(
        result_path='results/train_results.csv',
        test_results='results/test_results.csv',
        output_path='results/plots/losses.png'
        )
    # plot(
    #     result_path='results/test_results.csv',
    #     output_path='results/plots/test_accuracy.png'
    #     )