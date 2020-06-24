# NDADM Summary

---

- Solution: BellCNN

- Dataset: Kaggle
  (Source: https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images/data)

  class: 4 classes

  image size: 176*208

---

- Optimizer: Adam

- Activation function: Softmax

- BellCNN

  1. Prepare input layer as initial layer
  2. Set input of layer N, to output of layer N-1

  3. Set stride to 5

  4. Set activation to RELU
  5. Convolve with 32 filters
  6. Max pool
  7. Convolve with 64 filters
  8. Max pool
  9. Convolve with 128 filters
  10. Max pool
  11. Convolve with 64 filters
  12. Max pool
  13. Convolve with 32 filters
  14. Max pool
  15. Flatten with Fully connected layer of 1344nodes
  16. Set dropout rate to 0.8
  17. Set activation to SoftMax
  18. Output with final layer of 4 nodes
  19. Set optimizer to Adam
  20. Regress

