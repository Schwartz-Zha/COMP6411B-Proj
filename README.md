# Done:

1. tinyImageNet Dataloader and CIFAR10 Dataloader
2. Low-resolution image classifier training with VGG16-BN
3. Super-resolution network training with SRResNet (scaling factor = 4)
4. Separately train super-resolution network and classifier
5. Jointly train super-resolution network and classifier

# TODO:

- Dataloader of different datasets (Refer to dataloader of tinyImageNet in dataset.py)

## 1. Train low-resolution image classifier

### 1.1 tiny ImageNet

```python
python train_cls.py --device 1 --net vgg16 --dataset tinyImageNet --num_epoch 100 --lr 0.01
```

*Note: Experiments indicate SGD with momentum performs better than Adam on tinyImageNet.*

**Performance on original dataset (HR dataset, 50 epochs):**

Train accuracy: 99%; test accuracy after 50 epochs training: 50.37%.

**Performance on Low-resolution dataset (100 epochs):**

Train accuracy: 99%; test accuracy after 100 epochs: 25.71%.

### 1.2 CIFAR10

```python
python train_hr.py --device 1 --net vgg16 --dataset CIFAR10 --num_epoch 50 --lr 0.01
```

**Performance on original dataset (HR dataset, 50 epochs):**

Train accuracy: 99.99%; test accuracy after 100 epochs: 88.56%.

```python
python train_cls.py --device 1 --net vgg16 --dataset CIFAR10 --num_epoch 100 --lr 0.01
```

**Performance on Low-resolution dataset (100 epochs):**

Train accuracy: 99.99%; test accuracy after 100 epochs: 67.44%.

## 2. Separately train super-resolution network and classifier

### 2.1 Train super-resolution network(SRResNet) 

#### 2.1.1 tinyImageNet

```python
python train_sr.py --device 1 --dataset tinyImageNet --num_epoch 100 --lr 0.01
```

The performance of SR network is measured by MSE loss and PSNR (Implemented in train_sr.py).

**Performance (100 epochs):**

Test MSE loss: 0.0464; test PSNR: 20.23.

#### 2.1.2 CIFAR10

```python
python train_sr.py --device 1 --dataset CIFAR10 --num_epoch 100 --lr 0.01
```

The performance of SR network is measured by MSE loss and PSNR (Implemented in train_sr.py).

**Performance (100 epochs):**

Test MSE loss: 0.0336; test PSNR: 21.38.

### 2.2 Train classifier with super-resolution results

#### 2.2.1 tinyImageNet

```python
python main_sep.py --device 1 --dataset tinyImageNet --num_epoch 50 --lr 0.01
```

**Performance (50 epochs):**

Train accuracy: 99%; test accuracy: 34.72%.

#### 2.2.2 CIFAR10

```python
python main_sep.py --device 1 --dataset CIFAR10 --num_epoch 50 --lr 0.01
```

**Performance (50 epochs):**

Train accuracy: 99.988%; test accuracy: 70.33%.

## 3. Jointly train super-resolution network and classifier

### 3.1 tinyImageNet

```python
python main_joint.py --device 1 --dataset tinyImageNet --num_epoch 50 --lr 0.01 --sr_weight 0.1
```

**Performance (50 epochs):**

Training accuracy: 99%; test accuracy: 35.24%; test PSNR: 17.55%.

***Note: The current learning rate scheduler (Multi step LR) is not good. All the above results seem overfitting on trainset. Need more careful tuning.***

### 3.2 CIFAR10

```python
python main_joint.py --device 5 --dataset CIFAR10 --num_epoch 50 --lr 0.01 --sr_weight 0.5
```

**Performance (50 epochs):** 

Training accuracy: 99.988%; test accuracy: 70.21%; test PSNR:19.44

```python
python main_joint.py --device 5 --dataset CIFAR10 --num_epoch 50 --lr 0.01 --sr_weight 1
```

**Performance (50 epochs):** 

Training accuracy: 99.98%; test accuracy: 69.96%; test PSNR:20.18