
# Results: K-Means and Decision Trees

## Overview

### K-Means Clustering
- Preferred setup: **k = 2** with **elliptical mask**
- Main challenge: The nest often blends with visually similar areas (e.g., wall, sofa)
- Features like **softness** and **brightness** weren’t distinctive enough
- A two-step approach is necessary:
  - Use clustering to roughly locate the nest area
    2. Refine that cluster using a decision tree


### Decision Tree Classification
- Binary labeling: **Cat-nest = 1**, **Other = 0** performed well
- Three-class setup (*Cat-nest*, *Wall*, *Other*) underperformed
- Improvement with added **texture** feature, because features (**softness**, **brightness**) are not distinctive enough

---

## Detailed Explanation of the Results

### Clustering with K-Means

To begin, I applied K-Means clustering to segment the image into distinct regions. I experimented with various values for **k** and used different masking strategies to focus the algorithm:

- **Values of k** were varied to observe cluster behavior.
- **Different masks** were used to guide the clustering:
  - A **rectangular mask** includes the area around the nest.
  - An **elliptical mask** matches the nest's real shape and gives better results.
- **Softness** and **brightness** were calculated using multiple window sizes

The most promising outcome came from **k = 2** combined with the **elliptical mask**. Even so, the nest was not always cleanly separated: the wall and the sofa shared similar tones and textures, confusing the clustering.

### Refinement with Decision Tree

To improve segmentation after clustering, I trained a **decision tree classifier** on the clustered image. The binary classification (Cat-nest = 1, Other = 0) performed well, offering a cleaner separation than the K-Means clustering alone. However, the three-class setup (*Cat-nest*, *Wall*, *Other*) underperformed, as the features didn’t provide enough contrast to cleanly differentiate all three categories.

------

### Feature Engineering: Adding Texture

The initial features — **softness** and **brightness** — were useful but not distinctive enough in some areas, especially between similar-looking regions like the wall and the cat's nest. To address this, I introduced **texture** as an additional feature.

- **Softness**: Calculated using Laplacian edge detection, softness highlights areas with more texture. However, it didn’t provide enough distinction between the wall and the cat's nest.
- **Brightness**: Calculated using a 5x5 kernel on the grayscale image, brightness captures areas of warmth. This feature was dominant and was responsible for most of the decision-making. However, it was heavily dependent on the position of the window, which limited its utility.
- **Texture**: Calculated as the local standard deviation of pixel intensities. This helped capture structural variations, making it easier to differentiate between the smooth, flat wall and the more irregular, fluffy cat's nest or couch.

By adding texture, the model’s ability to separate the nest from other areas improved, where the initial features lacked enough clarity.
