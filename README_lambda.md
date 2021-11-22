## Install

```
git clone https://github.com/chuanli11/transformers.git
cd transformers

virtualenv -p /usr/bin/python3.8 venv-lambda
. venv-lambda/bin/activate

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

cd benchmark
pip install -r requirements.txt
```


## Usage

```
cd benchmark
python benchmark.py
```

