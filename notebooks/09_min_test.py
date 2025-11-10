import marimo

app = marimo.App()


@app.cell
def _(mo):
    mo.md("# Test cell")
    return


if __name__ == '__main__':
    app.run()
