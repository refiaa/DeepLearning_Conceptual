# DeepLearning_Conceptual

> Fuckin compiler error wont shut up in `./src/Model/cnn_model.py` and `./src/Model/lstm_model.py`
>
> That is why there is fuckin `# type: ignore` in every tensorflow.keras import

Somehow, things have been miraculously working.

I'm not sure why...

Below here is Conceptual Structure of each DeepLearning model (I know LightGBM is not a DeepLearning model but CNN and LSTM are kinda DeepLearning models, let's be cool with this)

also i use `Optuna` for hyperparameter optimization, and you can split up the training and validation periods (like with `validation_split` in yml and all that).

The validation period stuff gets saved as a CSV too.

```
These codes are actually functional, but I've edited some parts as they're intended for conceptual understanding.

You can make these codes fully operational by fixing them yourself and obtaining your own datasets.
```

---

## CNN
![Layer CNN](./img/layer_cnn.svg)

## LSTM
![Layer LSTM](./img/layer_lstm.svg)

*images are also conceptual so actually there is some error.im too lazy to fix it*
