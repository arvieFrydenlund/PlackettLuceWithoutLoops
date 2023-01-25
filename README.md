A GPU efficient vectorized Plackett-Luce loss in PyTorch from [Language Modelling via Learning to Rank, Frydenlund et al. AAAI22.](https://arxiv.org/abs/2110.06961)

1) This is tested for partial Plackett-Luce (k known ranks of v total classes), it is untested for full rankings.
2) This includes the biased partitioned preference modification, which treats all items in a partition (order) as equally preferred.

See the testing or training scripts for usage.

***Please cite when using this code*** 

`@inproceedings{frydenlund2022language,
  title={Language Modelling via Learning to Rank},
  author={Frydenlund, Arvid and Singh, Gagandeep and Rudzicz, Frank},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={10636--10644},
  year={2022}
}`


