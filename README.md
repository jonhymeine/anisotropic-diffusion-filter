# Anistropic Diffusion Filter

This is a simple implementation of the anisotropic diffusion filter in Python. Then, it's compared to the Gaussian filter, applying both to a noisy image and measured with the MSE and PSNR metrics.

It's also applied to the images:
- the Sobel filter to show the edges;
- the Unsharp Masking and Highboost Filtering to enhance the edges.
  - and then, the Sobel filter again to show the enhanced edges.

## How to run

It's possible to run directly with Python installed or with a Dev Container.

**Directly with Python:**

```bash
pip install -r requirements.txt
python app/main.py
```

**With Dev Container**:

It's necessary to have Docker installed and the VS Code Dev Container extension. Then, open the project in Visual Studio Code and click on the "Reopen in Container" button.

