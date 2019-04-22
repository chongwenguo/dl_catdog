# Classification for Cats and Dogs

## Classification Problems
1. Binary Classification: cat vs dogs
2. Breed Classification for cats
3. Breed Classification for dogs

## Problem Setup
For each problem, we construct two models:
* Build from Scratch
* Fine-Tune Pre-trained Models

For each model, we experiment it with various data augmentation techniques:
* No augmentation
* Traditional data augmentation techniques
* Data Augmentation with GAN

Sample Experimental Results Table:<br/>
<br/>

|                    | No Augmentation | Augmentation 1 | GAN |
|--------------------|-----------------|----------------|-----|
| Model from Scratch |      0.3737     |       0.3758   | N/A |
| Resnet from Scratch|      0.4168     |       0.4536   | N/A |
| Fine-tuned Model   |      0.7797     |       0.8035   | N/A |
