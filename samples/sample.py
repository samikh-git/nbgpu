# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch", "numpy"]
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
    return (torch,)


@app.cell
def _():
    1+1 == 2
    return


@app.cell
def _():
    3 + 3
    return


@app.cell
def _(torch):
    torch.cuda.is_available()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is going on here?

    $1 + 1$
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
