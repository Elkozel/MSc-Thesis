# Description

This is the implementation for the paper named "An Intrusion Detection System using Graph Neural Networks". 

# For researchers

Dear researcher, welcome to my implementation, hopefully it is readable and clear enough! If you have any questions, please feel free to send me an email!

To understand this project fully, you will need to know how PyTorch works, how to create a model in PyTorch, and the basics of PyTorch Lightning. All our models are implemented as [LightningModules](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and the datasets used are implemeted as [LightningDataModules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

I would also advice you to take a look at the paper, as it tries to explain some of the models in the "Implementation" section.

## Running
Please refer to the file named `run.py` to check how to run each model and each dataset. A simple code example, without all the argument parsing would look like this:
```js
comet_logger = CometLogger(
    api_key="",
    project="",
    workspace=""
)
dataset_folder = "/data"
dataset = UWF22(dataset_folder)
model = Try1(
    dataset.node_features + 2,
    dataset.node_features + 2 * 3
    )

trainer = L.Trainer(logger=comet_logger)
trainer.fit(model, dataset)
trainer.test(model, dataset)
```

## Logging
This project used the built in loggin in PyTorch Lightning (with a logger for the Comet platform). If you would like to change the loggin, please refer to the [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

There might be one issue, since the logger is called to log the confusion matrix, I would recomment you try it with your preferred loggin method, and if it fails, delete the line responsible for the loggin of the confusion matrix (in the `Try1` file): 
```python
    def on_test_end(self):
        # Upload the confusion matrix
        matrix = self.link_class_matrix.compute().cpu().numpy()
        self.logger.experiment.log_confusion_matrix(matrix=matrix)
```
this will not impact any of the other functions.