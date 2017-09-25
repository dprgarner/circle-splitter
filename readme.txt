An attempt at solving this circle-splitting problem: https://www.reddit.com/r/dailyprogrammer/comments/6ksmh5/20170630_challenge_321_hard_circle_splitter/

To run an example:
```sh
$ python circlesplitter.py < eg1.txt
```
or:
```sh
$ python circlesplitter.py eg1.txt
``` 

To draw the calculated circle on a plot, create a virtualenv, install matplotlib, and set the keyword arg `plot=True` when calling `CircleSplitterParser`.
