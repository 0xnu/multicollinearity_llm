## multicollinearity_llm

### Understanding Multicollinearity

Upon encountering the term `multicollinearity`, I decided to explore its definition to grasp its significance. Let's explore this concept in depth.

#### What is Multicollinearity?

[Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) refers to a phenomenon in statistical analysis where two or more explanatory variables in a multiple regression model are highly correlated. In essence, it describes a situation where there exists a strong linear relationship between predictor variables.

To put it simply:

* Multicollinearity occurs when independent variables in a model are not truly independent of each other.
* The absence of multicollinearity implies that no substantial linear relationship exists between the explanatory variables.

#### Mathematical Expression

We can express the assumption of no multicollinearity mathematically as follows:

For any combination of coefficients a₀, ..., aₖ (not all zero) and k > 0:

$$ E[a_0 + a_1X_1 + ... + a_kX_k] > 0 $$

Where E denotes the expected value, and X₁, ..., Xₖ are the explanatory variables.

#### An Illustrative Example

To better understand multicollinearity, consider a scenario where identical variables appear multiple times in a regression model:

$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + u $$

In this model, if X₁ = X₂ always holds true, then by choosing appropriate values for a₁ and a₂, we can create conditions that violate the previous assumption of no multicollinearity.

#### Implications of Multicollinearity

* Unstable coefficient estimates
* Inflated standard errors
* Difficulty in determining individual variable importance
* Potential overfitting of the model

#### Detecting Multicollinearity

Researchers often use methods such as:
* Variance Inflation Factor (VIF)
* Correlation matrices
* Eigenvalue analysis

#### Addressing Multicollinearity

When faced with multicollinearity, analysts might:
* Remove one of the correlated variables
* Combine correlated variables
* Use regularisation techniques (e.g., ridge regression)
* Collect more data, if possible

Understanding and addressing multicollinearity is compulsory for building statistical models and drawing accurate conclusions from data analysis.

#### Large Language Models (LLMs) Multicollinearity-based Compression

The core idea centres on weights (or neurons) exhibiting high correlation with one another, which likely contain redundant information. Reducing multicollinearity in the model can be achieved by removing these correlated weights, resulting in a more effective and potentially more generalisable model.

Let's define the compression algorithm mathematically:

For a given layer $L$ with weight matrix $W \in \mathbb{R}^{m \times n}$, where $m$ is the number of neurons in the current layer and $n$ is the number of neurons in the previous layer:

+ Define a correlation function $C(w_i, w_j)$ between two weight vectors $w_i$ and $w_j$:

   $$C(w_i, w_j) = \frac{\sum_{k=1}^n (w_{ik} - \bar{w_i})(w_{jk} - \bar{w_j})}{\sqrt{\sum_{k=1}^n (w_{ik} - \bar{w_i})^2} \sqrt{\sum_{k=1}^n (w_{jk} - \bar{w_j})^2}}$$

   where $\bar{w_i}$ and $\bar{w_j}$ are the means of $w_i$ and $w_j$ respectively.

+ Define a pruning indicator function $P(w_i)$ for each weight vector $w_i$:

   $$P(w_i) = \begin{cases}
   1 & \text{if } \max_{j < i} |C(w_i, w_j)| \leq \tau \\
   0 & \text{otherwise }
   \end{cases}$$

   where $\tau$ is the correlation threshold.

+ The compressed weight matrix $W'$ is then defined as:

   $$W' = \{w_i : P(w_i) = 1, i = 1, ..., m\}$$

+ The compression ratio $R$ for the layer is given by:

   $$R = 1 - \frac{|W'|}{|W|}$$

   where $|W|$ and $|W'|$ are the number of weights in the original and compressed matrices respectively.

+ The overall model compression is achieved by applying this process to all layers:

   $$M' = \{L'_1, L'_2, ..., L'_k\}$$

   where $L'_i$ is the compressed version of the $i$-th layer, and $k$ is the total number of layers.

This approach to addressing multicollinearity is based on model pruning, one of several techniques used in model compression. Other methods in the code, like magnitude-based or variance-based pruning, also address multicollinearity by removing less important weights, which may be correlated with more important ones.

To fully utilize this multicollinearity-based compression, the initial weight initialization should provide sufficient variability and the correlation threshold $tau$ should be set appropriately for your specific use case.

#### Usage and Extension

To use it in your project, compile the library using the [Makefile](./Makefile), link against the resulting `lib/libmodelcompressor.a`, and include `include/model_compressor.h` in your source files. Finally, after training your model, apply the compression functions to the trained weights.

This implementation provides a foundation for model compression. You may extend it with more advanced pruning or distillation techniques as needed. For large language models, consider fine-tuning the compressed model to adapt it to specific tasks or domains. Doing so could improve its performance.

#### References

- [Enlarging of the sample to address multicollinearity](https://arxiv.org/abs/2407.01172)
- [Covariance Matrix Analysis for Optimal Portfolio Selection](https://arxiv.org/abs/2407.08748)
- [Explainable Artificial Intelligence and Multicollinearity : A Mini Review of Current Approaches](https://arxiv.org/abs/2406.11524)
- [Regularized boosting with an increasing coefficient magnitude stop criterion as meta-learner in hyperparameter optimization stacking ensemble](https://arxiv.org/abs/2402.01379)

#### License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).

#### Citation

```tex
@misc{mcllm2024,
  author       = {Oketunji, A.F.},
  title        = {Understanding Multicollinearity},
  year         = 2024,
  version      = {0.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13308667},
  url          = {https://doi.org/10.5281/zenodo.13308667}
}
```

#### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.