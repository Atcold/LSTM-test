# LSTM test

To run the code type:

```bash
th -i LSTM-hw.lua
```

To output the weight of a `nn.Linear()` layer: check on the SVG which node you want. Then:

```lua
myNode = 20
getParameters(LSTM_module, myNode)
```

To print the output of the `LSTM_module`:

```lua
print(outTable)
```
