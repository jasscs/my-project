# Jasmin Francisco - Projects

A collection of the projects I've built, ideas I've explored, and experiments I've worked on in computer vision and machine learning.

---

## Lab Activity 1: Coin Detection

### Overview
This project applies **convolution, spatial filtering, and edge detection** to images of Philippine coins from the provided dataset. The goal is to produce a clean edge-detected result in which the coins that make up **₱21.10** (₱20 + ₱1 + ₱0.10) can be clearly identified.

### Dataset
Located in the `coin_images/` folder:

| File | Coin |
|------|------|
| `twenty_php.jpg` | ₱20 |
| `piso_php.jpeg` | ₱1 |
| `ten_centavo.jpeg` | 10 centavos |

### Methodology

**1. Preprocessing (spatial filtering)**
Each image is converted to grayscale, then a 3x3 median filter removes noise and a Gaussian blur (radius 1) smooths the image.

**2. Convolution with a sharpening kernel**
A 3x3 sharpening kernel is applied using a convolution function written from scratch. The kernel slides across the image one pixel at a time. At each position, the surrounding pixel values are multiplied by the kernel and summed. This increases the contrast between neighboring pixels so the coin details and markings stand out before edge detection.

```
[ 0, -1,  0]
[-1,  5, -1]
[ 0, -1,  0]
```

**3. Edge detection (Sobel)**
Sobel operators detect strong intensity changes in the horizontal and vertical directions. The two results are combined into one edge map, which makes the circular coin boundary and engraved details easier to identify.

**4. Post-processing**
The raw edge map is cleaned up through:
- Median smoothing to reduce noise
- Normalization and histogram equalization to improve contrast
- Binary thresholding to isolate the edges
- Morphological closing and opening to fill small gaps and remove specks

The final result keeps the circular outline clear and preserves the visible coin markings. An overlay on the grayscale image is used for verification.

### Techniques Used
Median filtering, Gaussian blur, 2D convolution, image sharpening, Sobel edge detection, histogram equalization, thresholding, morphological operations

### Libraries
`numpy`, `opencv-python`, `matplotlib`, `Pillow`

### Files
- `LA1_FRANCISCO.ipynb`: main notebook
- `coin_images/`: dataset

---

## Lab Activity 2: Can a Simple Neural Network Recognize Images?

### Overview
This project builds a simple fully connected neural network in TensorFlow/Keras to classify images of rice grains into five varieties. It also investigates how image resolution affects accuracy and training time.

### Dataset
**Rice Image Dataset**, with 75,000 images in total:

| Class | Images |
|-------|--------|
| Arborio | 15,000 |
| Basmati | 15,000 |
| Ipsala | 15,000 |
| Jasmine | 15,000 |
| Karacadag | 15,000 |

The dataset is perfectly balanced, and all images are RGB (3 channels). It is not included in this repository because of its size. Download it separately and set `DATASET_DIR` in the notebook to its location.

### Methodology

**1. Data preparation**
- Images are resized to 32x32 and pixel values are rescaled from 0-255 to 0-1
- The data is split into **70% training** (52,500 images), **15% validation**, and **15% test** (22,500 images split evenly between validation and test)

**2. Model architecture**
A simple feed-forward network with 820,229 trainable parameters:

| Layer | Output Shape |
|-------|--------------|
| Flatten | 3072 |
| Dense (ReLU) | 256 |
| Dense (ReLU) | 128 |
| Dense (Softmax) | 5 |

**3. Training**
- Optimizer: Adam
- Loss: Sparse categorical crossentropy
- Epochs: 3

**4. Evaluation**
The model is evaluated using test accuracy, a classification report (precision, recall, F1-score), a confusion matrix, and visual inspection of correct and incorrect predictions.

### Results

**Initial run (batch size 8):** 97.64% validation accuracy and **95.62% test accuracy**. Per-class F1-scores ranged from 0.94 to 0.99, with Ipsala the easiest to classify (0.99) and Basmati and Jasmine the hardest (0.94).

**Final run (batch size 32):** 10,968 of 11,264 test images were classified correctly (about **97.4% accuracy**), with 296 misclassified.

**Resolution comparison:**

| Image Resolution | Test Accuracy | Training Time |
|------------------|---------------|---------------|
| 32x32 | 0.9752 | 38.8 min |
| 64x64 | 0.9691 | 24.8 min |
| 128x128 | 0.9668 | 47.3 min |

The simple network performed best at the smallest resolution, 32x32. Higher resolutions did not improve accuracy, since a fully connected network has many more weights to learn as the input grows.

**Common mistakes:** The model most often confused Jasmine with Basmati, and Karacadag with Arborio, which suggests these varieties look similar at low resolution.

### Techniques Used
Image normalization, train/validation/test splitting, fully connected neural network, softmax classification, Adam optimization, confusion matrix analysis

### Libraries
`tensorflow`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `Pillow`

### Files
- `LA2_Francisco.ipynb`: main notebook

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/jasscs/jasminfrancisco.git
   ```
2. Install the required libraries:
   ```bash
   pip install numpy opencv-python matplotlib pillow tensorflow scikit-learn seaborn
   ```
3. For Lab Activity 2, download the Rice Image Dataset and update `DATASET_DIR` in the notebook
4. Open the notebooks in Jupyter Notebook or Google Colab
5. Run the cells from top to bottom

---

## Author
**Jasmin Francisco**
