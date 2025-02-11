from nltk.parse import CoreNLPParser
from sys import argv

parser = CoreNLPParser(url='http://localhost:9000')


# sentence string as our argument
sent = argv[1]
analysis = list(parser.raw_parse(sent))
analysis[0].pretty_print()


# example 
p = list(parser.raw_parse('What had Theresa walked through while talking about that high school?'))
# p[0].pretty_print()

# parentheses notation of the tree  
paren_str = ' '.join(str(p[0]).split())

