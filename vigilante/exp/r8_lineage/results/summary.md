# R8 same-family prediction vs detection

Accuracy (k/n correct, Wilson 95%); unparsed answers count as wrong.

| judge | family | predict | detect | predict: says Yes | unparsed |
|---|---|---|---|---|---|
| qwen36 | self | 52/80 (65%) [54%,75%] | 79/80 (99%) [93%,100%] | 30/80 | 0 |
| qwen38 | same family | 40/80 (50%) [39%,61%] | 59/80 (74%) [63%,82%] | 0/80 | 0 |
| gemma4 | other | 45/80 (56%) [45%,67%] | 73/80 (91%) [83%,96%] | 7/80 | 0 |
| sonnet | other | 41/80 (51%) [40%,62%] | 80/80 (100%) [95%,100%] | 3/80 | 0 |
| gpt | other | 40/80 (50%) [39%,61%] | 79/80 (99%) [93%,100%] | 0/80 | 0 |
