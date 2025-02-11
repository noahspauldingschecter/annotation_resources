from nltk import Tree 
from sys import argv

# make sure you've installed nltk (https://www.nltk.org/install.html)

parens_str = argv[1]
t = Tree.fromstring(parens_str)
t.pretty_print()



# example 
t = Tree.fromstring("((ROOT (SBARQ (WHNP (WP who)) (SQ (VP (MD should) (NP (NNP Derek)) (VP (VB hug) (NP (-NONE- *T*)) (SBAR (COMP after) (S (VP (VBG shocking) (NP (NNP Richard)))))))) (. ?))))")
# t.pretty_print()