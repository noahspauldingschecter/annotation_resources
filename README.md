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

<pre> ```bash cd stanford-corenlp-4.5.8 java -mx4g -cp "*" 
edu.stanford.nlp.pipeline.StanfordCoreNLPServer \ -preload 
tokenize,ssplit,pos,lemma,ner,parse,depparse \ -status_port 9000 -port 9000 
-timeout 15000 &``` </pre>

This creates a local server where the downloaded parser model is loaded 
into memory allowing us to send requests through our python code. 

Then, from the terminal, we can run our parsing file which will connect to 
the local server you've just established. Download parsing.py and navigate 
to the directory it is in. 

<pre> ```bash python parsing.py "Here is an example sentence."``` </pre>

This runs pasrsing.py with the argument "Here is an example sentence." which
the file takes in and parses. You should see a tree printed in your terminal.
Make sure to include quotation marks! 

Finally, the server will be running in the background until you kill it. I use
the following command to bring the process to the foreground and then press
"control c" to stop it.

<pre> ```bash fg Ctrl + C``` </pre>  

### Tree from parentheses notation
To display a tree structure from parentheses notation, you can download the 
tree_structure_from_string.py python file and run the following command from
the terminal.

<pre> ```bash python tree_structure_from_string.py 
"(S (NP (N Mary)) (VP (V slept)))"``` </pre>

This takes the string in quotation marks as an argument. You can replace this 
string to visualize more trees. 
