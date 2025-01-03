# Face-Mask-detection[ANN-CNN-Transfer-Learning]

I used three different classification approaches in an image in a mask detection task; namely, classifiers for maximum accuracy in terms of distinguishing between categories.

Artificial Neural Network (ANN)
Achieved Test Accuracy 0.75 (75%) Architecture: A Deep ANN with many fully connected layers and regularization techniques like dropout and L2 regularization. Observations ANN was able to capture simple patterns but not the spatial features of the images. The flatting into a 1D vector of the image data, which is spatially 2D, might explain the limitations in accuracy. Despite achieving accuracy of 75%, the ANN failed to fully exploit inherent spatial information in image data.

Convolutional Neural Network (CNN)
Test Accuracy Reached: 0.75 (75%) Architecture: Single simple CNN architecture with multiple convolutional layers, pooling, batch normalization, and dropout. Observations: So, CNNs have been designed natively better with handling the image data, because convolutional layers keep spatial relationships intact. Although the model applied convolutional filters, accuracy still had a stalemate at 75%, suggesting that further tuning or capacity of the model might be needed. Though similar in performance to ANN, CNN was able to attain this by performing better at feature extraction and thus better at the choice of model for image data.

Transfer Learning (Pre-trained Model)
Test Accuracy Achieved 0.80 80% Architecture: Transfer Learning of a pre-trained model: in this case, MobileNetV2, and appending custom dense layers for classification. Observations: Transfer learning was the best accuracy at 80% and performed better than both ANN and CNN. The pre-trained model worked very effectively to leverage features learned from big datasets such as ImageNet, thereby giving a marked performance boost. This can be learned with considerably greater efficiency and much faster than fine-tuning a pre-trained network. Transfer learning emerged to be the best performer in terms of both accuracy and training efficiency, through robust feature extraction from a pre-trained model. Comparison Overall ModelitectureTest Accuracy\tKey Strengths\tKey Limitations ANN Fully connected layers 0.75 Simple, straightforward to implement Struggles with spatial features CNNConvolutional + pooling layers0.75Capture spatial features; more suitable for imagesTuning is necessary to optimize performance Transfer Learning Pre-trained + custom layers 0.80 Leverage pre-trained features, faster training Dependent on quality of pre-trained weights

Key Takeaways

ANN: This would classify generally, but not for image-based content and its spatial relationship. Accuracy was limited despite increasing complexity, indicating that ANNs are not the best choice for image-based tasks.

CNN: Specially designed for images and has achieved similar performance to ANN but in a much more robust way. Although the CNNs may produce a better discernment of subtlety in images, more fine-tuning and possibly complexity may be required to attain stronger accuracy.

Transfer Learning: Provided the best results with minimal training time.

Benefited from features learned on large-scale datasets, making it more effective at handling image complexities.

It gave an outstanding accuracy of 5% over ANN and CNN; hence, it is the best approach used so far.

Future Recommendations for Improvement

More Fine-Tuning In transfer learning, fine-tune deeper layers of the pre-trained model with the data to really capture features specific to the dataset.

Ensemble Methods: Combine the predictions from multiple models with ANN, CNN, and even transfer learning to perhaps fine-tune accuracy.

Data Augmentation: If the data augmentation hasn't been applied, consider increasing it for a further generalization and increase diversity in the dataset.

Tuning Hyperparameters: Hyperparameter search (GridSearchCV and RandomSearch) should be performed to find the optimal set of learning rates, dropout rates, and batch sizes.

We can try other pre-trained models, like ResNet, InceptionV3, and EfficientNet, to check if any of them happen to be better.

Conclusion:

The best strategy through this project was Transfer Learning, as it showed the power and potency of pre-trained models in image classification. ANN provided a great starting point for image analysis but might need to use more advanced configurations and data post-adjustments. ANN, although easier to implement, turned out pretty weak with the image data, so indeed, specialized architectures (like CNN or Ttansfer learning) are more suited for such tasks.
