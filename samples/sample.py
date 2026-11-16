# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "torch",
#     "numpy",
#     "pillow",
# ]
#
# [tool.nbgpu]
# provider = "modal"
# gpu = "A10"
# idle-timeout = "20m"
# max-runtime = "2h"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import torch
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    return


@app.cell
def _():
    from PIL import Image

    return


@app.cell
def _():
    from PIL import Image

    return


if __name__ == "__main__":
    app.run()
