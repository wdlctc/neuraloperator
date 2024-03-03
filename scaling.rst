Scaling TFNO Training with Fully Sharded Data Parallel (FSDP)
==============================================================

In this tutorial, we will demonstrate how to efficiently scale the training of Temporal Fourier Neural Operators (TFNO) using Fully Sharded Data Parallel (FSDP) in PyTorch. This approach is particularly useful for training large models that do not fit into a single GPU memory.

Prerequisites
-------------

Before we begin, ensure that you have the following installed:

- PyTorch
- Matplotlib
- The ``neuraloperator`` library

You can install the required libraries using pip:

.. code-block:: bash

    pip install torch matplotlib neuraloperator

Setup
-----

First, let's import the necessary libraries:

.. code-block:: python

    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import sys
    from neuralop.models import TFNO
    from neuralop import Trainer
    from neuralop.datasets import load_darcy_flow_small
    from neuralop.utils import count_model_params
    from neuralop import LpLoss, H1Loss
    import torch.multiprocessing as mp
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel
    import argparse

Distributed Environment Initialization
--------------------------------------

To utilize FSDP, we must initialize a distributed environment. The following function `benchmark` demonstrates how to set up the environment and distribute the model across multiple GPUs:

.. code-block:: python

    def benchmark(rank, args, world_size):
        device = 'cuda'
        RPC_PORT = 29501
        init_method_pgroup = f"tcp://localhost:{RPC_PORT}"
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
        )
        torch.cuda.set_device(rank)

Model and Dataset Preparation
-----------------------------

Next, we load the Darcy Flow dataset and initialize our TFNO model:

.. code-block:: python

        train_loader, test_loaders, data_processor = load_darcy_flow_small(
                n_train=10, batch_size=32,
                test_resolutions=[16, 32], n_tests=[100, 50],
                test_batch_sizes=[32, 32],
                positional_encoding=True
        )
        data_processor = data_processor.to(device)
        
        model = TFNO(n_modes=(64, 64), hidden_channels=256, projection_channels=512, factorization='tucker', rank=0.42)
        model = model.to(device)

FSDP Configuration
------------------

We then wrap our model with FSDP to shard the model parameters:

.. code-block:: python

        fsdp_params = dict(mixed_precision=True, flatten_parameters=True)
        model = FullyShardedDataParallel(model)

Training
--------

After configuring our model with FSDP, we can proceed to train it:

.. code-block:: python

        optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)
        
        train_loss = h1loss
        eval_losses = {'h1': h1loss, 'l2': l2loss}
        
        trainer = Trainer(model=model, n_epochs=20, device=device, data_processor=data_processor, use_distributed=True, verbose=True)
        
        trainer.train(train_loader=train_loader, test_loaders=test_loaders, optimizer=optimizer, scheduler=scheduler, training_loss=train_loss, eval_losses=eval_losses)

Visualization
-------------

Finally, we visualize the results:

.. code-block:: python

        test_samples = test_loaders[32].dataset
        # Visualization code here...

Main Function
-------------

To run the distributed training, use the following main function:

.. code-block:: python

    def parse_args():
        parser = argparse.ArgumentParser(description="benchmark")
        parser.add_argument("--max_batch", type=int, default=4, help="Max number of batches")

    if __name__ == "__main__":
        args = parse_args()
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        mp.spawn(
            benchmark,
            args=(args, num_devices),
            nprocs=num_devices,
            join=True,
        )

Conclusion
----------

This tutorial has demonstrated how to scale the training of TFNO models using PyTorch's FSDP. By sharding model parameters across multiple GPUs, we can train larger models that would otherwise not fit in the memory of a single GPU.
