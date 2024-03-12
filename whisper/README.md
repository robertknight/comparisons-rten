## Quick start

Download [weights](https://www.dropbox.com/scl/fi/8efloh7muvcevgq1sqd7f/weights.zip?rlkey=bkcnd2la49bx0uqfarv1nzp28&dl=1) and [example audio file](https://www.dropbox.com/scl/fi/8yzo8y2ptxoy0rfuon9bu/audio.wav?rlkey=dorb43edb48bqpx5cgrtckxlk&dl=1). Unzip weights.zip to whisper directory, create "data" directory and put audio there.

### Run rust code:

```
cd whisper-rs

cargo run --release
```

### Run python code:

```
cd whisper-py

python3 main.py
```

### Run python code with quant models:

```
cd whisper-py

python3 main.py --quant
```
