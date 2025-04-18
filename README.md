# Update 4/7/25
I have added files for the non-wugged BLiMP test sets. This collection
is all the test items for which the learned grammar has seen all the
lexical items during training. I have one collection "all_samples" that
contains all the items that all of the samples can be tested on without
adding the special unseen character <WUG>. Then I have a collection for
each sample. 

# Annotation Resources
I have compiled some resources for performing syntactic annotation.
Ultimately, we want to create trees for sentences in line with the CHILDES
Treebank annotation. We'll start with the good and bad sentences in the 
"adjunct_island.jsonl" file. I have started with the first 10 in 
"adjunct_island_trees.txt." 

For resources, I have included: 
- trees from the brown adam corpus that can be used as a reference 
(ExampleTrees_brown_adam.txt) 
- some relevant annotation notes from the CHILDES Treebank readme file 
(Treebank_annotation_notes.txt) 
- the list of labels with a description of each (hosted on this webpage: 
[Labels](https://nielswd23.github.io/annotation_resources/)) 


## Coding Resources 
I have also included two python files. Both require the python NLTK package. 
Here are the instructions for downloading it 
[link](https://www.nltk.org/install.html).

### Parsing
parsing.py connects to the Stanford 
parser and prints out a parsed tree for the input sentence. This relies on 
the relevant parser code being downloaded locally.    

First download Stanford's CoreNLP [link](https://stanfordnlp.github.io/CoreNLP/).
When you unzip the file you will get a folder "stanford-corenlp-4.5.8" 
depending on the version number. Now in the terminal, navigate to this 
directory and start the parser.

<pre>cd stanford-corenlp-4.5.8</pre> 
<pre>java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \ 
-status_port 9000 -port 9000 -timeout 15000 & </pre>

This creates a local server where the downloaded parser model is loaded 
into memory allowing us to send requests through our python code. The second 
command starts with "java" and goes to "&", so copy the whole thing and paste
it in the terminal. The backslash just denotes that command is not finished.

Then, from the terminal, we can run our parsing file which will connect to 
the local server you've just established. Download parsing.py and navigate 
to the directory it is in. 

<pre>python parsing.py </pre>

This program will prompt you to type in the sentence you want to parse and the 
format you want the output to be (tree format, parentheses format, or both). 

Finally, the server will be running in the background until you kill it. I use
the following command to bring the process to the foreground and then press
"control c" to stop it.

<pre>fg</pre> 
<pre>Ctrl + C </pre>  

### Tree from parentheses notation
To display a tree structure from parentheses notation, you can download the 
tree_structure_from_string.py python file and run the following command from
the terminal.

<pre>python tree_structure_from_string.py "(S (NP (N Mary)) (VP (V slept)))" </pre>

This takes the string in quotation marks as an argument. You can replace this 
string to visualize more trees. 
