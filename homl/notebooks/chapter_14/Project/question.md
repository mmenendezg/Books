Hello, 

I am training an image classification model using [Resnet-50 V2](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5) trained on imagenet as base of the model. I am using Tensorflow 2.11 and the model comes from Tensorflow hub.

The dataset to classify is the [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/code) from Kaggle. The original size of the image is `768x768`. Before being fed to the model, the images are preprocessed. The pipeline is the following:

- The images are separated into training, validation and testing sets and stored as TFRecord files
- The images are loaded and resized to `448x448`.
- Pixel values are standardized to be between 0 and 1 (as indicated in tensorflow hub documentation)
- The datasets are batched, training set is shuffled and all are cached in memory (dataset size is around 2GB).
- Datasets are prefetched to reduce training time. 
- There is no data augmentation applied since the data has been already augmented.
- The first layer of the model is a `MaxPool2D` layer to decrease the size of the image to `224x224` as recomended in the documentation.

You can find the full code in my [GitHub](https://github.com/mmenendezg/Books/blob/homl/homl/notebooks/chapter_14/Project/lung_colon_histopathology.ipynb).

When I try to train the model, the notebook fails and shows the following error:

> The Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure. Click here for more info. View Jupyter log for further details.

This happens before finishing the first epoch. First, I thought that it was the memory, so I reduced the size of the batch up to 4 images per batch. I checked memory usage with different batch sizes (from 4 to 128) and the memory usage in all cases is more or less the same.

Then, I tried to read the Jupyter logs, however they are not very descriptive about the cause of the crashing:

```
error 17:54:33.727: Error in waiting for cell to complete Error: Canceled future for execute_request message before replies were done
    at t.KernelShellFutureHandler.dispose (/Users/mmenendezg/.vscode/extensions/ms-toolsai.jupyter-2022.11.1003412109/out/extension.node.js:2:32353)
    at /Users/mmenendezg/.vscode/extensions/ms-toolsai.jupyter-2022.11.1003412109/out/extension.node.js:2:26572
    at Map.forEach (<anonymous>)
    at v._clearKernelState (/Users/mmenendezg/.vscode/extensions/ms-toolsai.jupyter-2022.11.1003412109/out/extension.node.js:2:26557)
    at /Users/mmenendezg/.vscode/extensions/ms-toolsai.jupyter-2022.11.1003412109/out/extension.node.js:2:29000
    at processTicksAndRejections (node:internal/process/task_queues:96:5)
warn 17:54:33.730: Cell completed with errors {
  message: 'Canceled future for execute_request message before replies were done'
}
info 17:54:33.732: Cancel all remaining cells true || Error || undefined
```

Find the full logs in my GitHub.

Could anyone help me to troubleshoot this situation. 

Additional details:

- GPU: M1 Pro
- Memory: 32GB
- Tensorflow 2.11
- Python 3.10.8
- Virtual environment using Poetry 1.2.2
