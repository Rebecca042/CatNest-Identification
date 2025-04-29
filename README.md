# CatNestIdentification
Project utilizing Decision Trees to refine K-means clustering for automatic cat nest identification.

Automating the identification of areas where a cat prefers to sleep presents several challenges. For example, the colors of the cat nest and surrounding objects and be very similar. Additionally, features like softness and brightness, which are typical of cat nests, can also be found in non-cat spots.

The idea is the following:
1.	Identify sleeping places for the cat manually.
2.	Use of AI to automate the identification of these spots.

Challenges:
-	Homogeneous colors of the wall, sofa, and cat nest.
-	Identification of features that will identify cat spots (e.g., softness, brightness).

Strategy:
-	K-means clustering with various optimizations:
    - Standard K-means
    - Initialization of the clusters to the cat nest area
    - Features softness and brightness (calculated with different window sizes)
-	Decision tree to refine the most relevant cluster:
    - Standard decision tree
    - Introduction of a ‘wall’ class
    - Features softness, brightness, and texture
