# Done:

1. tinyImageNet Dataloader
2. Low-resolution image classifier training with VGG16-BN
3. Super-resolution network training with SRResNet (scaling factor = 4)
4. Separately train super-resolution network and classifier
5. Jointly train super-resolution network and classifier

# TODO:

- Dataloader of different datasets (Refer to dataloader of tinyImageNet in dataset.py)

## Train low-resolution image classifier on tinyImageNet

```python
python train_cls.py --device 1 --net vgg16 --dataset tinyImageNet --num_epoch 100 --lr 0.01
```

*Note: Experiments indicate SGD with momentum performs better than Adam on tinyImageNet.*

**Performance on original dataset (HR dataset, 50 epochs):**

Train accuracy: 99%; test accuracy after 50 epochs training: 50.37%.

**Performance on Low-resolution dataset (100 epochs):**

Train accuracy: 99%; test accuracy after 100 epochs: 25.71%.

## Separately train super-resolution network and classifier

### 1. Train super-resolution network(SRResNet) on tinyImageNet

```python
python train_sr.py --device 1 --dataset tinyImageNet --num_epoch 100 --lr 0.01
```

The performance of SR network is measured by MSE loss and PSNR (Implemented in train_sr.py).

**Performance (100 epochs):**

Test MSE loss: 0.0464; test PSNR: 20.23.

### 2. Train classifier with super-resolution results

```python
python main_sep.py --device 1 --dataset tinyImageNet --num_epoch 50 --lr 0.01
```

**Performance (50 epochs):**

Train accuracy: 99%; test accuracy: 34.72%.

## Jointly train super-resolution network and classifier

```python
python main_joint.py --device 1 --dataset tinyImageNet --num_epoch 50 --lr 0.01 --sr_weight 0.1
```

**Performance (50 epochs):**

Training accuracy: 99%; test accuracy: 35.24%; test PSNR: 17.55%.

***Note: The current learning rate scheduler (Multi step LR) is not good. All the above results seem overfitting on trainset. Need more careful tuning.***