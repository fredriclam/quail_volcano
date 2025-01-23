#### Simple setup for testing code implementation.

Run `conduit.py` with the following commands. Depending on your installation of `python`, you may need to replace `python` with `python3` to use Python 3.

If quail is added to `$PATH` and `$PYTHONPATH`, navigate to this folder (`scenarios/simple_1D_test`) and execute:

```
quail conduit.py
```

If `quail` cannot be found, you can instead execute :

```
python <path/to/quail> conduit.py
```

#### Output

The output is a series of `.pkl` files, which are version-specific binary files that contain the solution object.