<h1 align="center">Baseline RecSys with Matrix Factorization </h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
    <a href="#" target="_blank">
    <img alt="License: APACHE" src="https://img.shields.io/github/license/francovm/recsys-mf-tf" />
  </a>
  <a href="#" target="_blank">
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/francovm/recsys-mf-tf" />
  </a>
  <a href="#" target="_blank">
    <img alt="Python version" src="https://img.shields.io/badge/python-%3E%3D3.7-blue" />
  </a>
</p>

> Simple Recommender System model via matrix factorization to build a strong baseline. 

## Install
From repository
```
pip install -e git+https://github.com/francovm/recsys-mf-tf.git#egg=recsys-mf
```

## Training

```python
from src.data import load_rating
from src.model import MFmodel, Recommender
import pandas as pd

data = load_rating(variant="100K")
df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])

model = MFmodel(regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=30, init_stddev=.05).fit(df)

model.train(num_iterations=2000, learning_rate=20.)

```

## Movie recommendation

```python
from src.model import MFmodel, Recommender, movie_neighbors, compute_scores

movie_neighbors(model, movies ,"Forrest Gump", cosine=False)
```
Output

```
	score	titles	genres
68	10.440863	Forrest Gump (1994)	Comedy-Romance-War
63	10.242167	Shawshank Redemption, The (1994)	Drama
317	10.217667	Schindler's List (1993)	Drama-War
173	9.372114	Raiders of the Lost Ark (1981)	Action-Adventure
422	9.129416	E.T. the Extra-Terrestrial (1982)	Children-Drama-Fantasy-Sci-Fi
97	9.091332	Silence of the Lambs, The (1991)	Drama-Thriller
```


## Author

👤 **Franco Vega **

* Website: LinkedIn
* Github: [@francovm](https://github.com/francovm)
* LinkedIn: [@franco-vega](https://www.linkedin.com/in/franco-vega/)






Give a ⭐️ if this project helped you!

