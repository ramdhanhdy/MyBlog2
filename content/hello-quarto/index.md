---
title: "Marketing Response: A not Comprehensive Analysis"
date: "2012-04-06"
tags: 
    - writing
format: 
    hugo-md:
        toc: true
jupyter: python3
---

-   <a href="#polar-axis" id="toc-polar-axis">Polar Axis</a>

## Polar Axis

For a demonstration of a line plot on a polar axis, see [Figure 1](#fig-polar).

<details>
<summary>Code</summary>

``` python
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
ax.set_rticks([0.5, 1, 1.5, 2])
ax.grid(True)
plt.show()
```

</details>

<figure>
<img src="index_files/figure-markdown_strict/fig-polar-output-1.png" id="fig-polar" width="450" height="439" alt="Figure 1: A line plot on a polar axis" />
<figcaption aria-hidden="true">Figure 1: A line plot on a polar axis</figcaption>
</figure>
