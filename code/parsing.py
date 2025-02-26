from nltk.parse import CoreNLPParser

# Set up the parser
parser = CoreNLPParser(url='http://localhost:9000')

def main():
    # Prompt user for sentence input
    sentence = input("Enter a sentence to parse: ")

    # Parse the sentence
    analysis = list(parser.raw_parse(sentence))

    # Keep prompting until a valid choice is given
    while True:
        format_choice = input("Choose output format - 't' for tree " + 
                            "visualization, 'p' for parentheses " + 
                            "notation, or 'b' for both: ").strip().lower()

        paren_str = ' '.join(str(analysis[0]).split())

        if format_choice == 't':
            analysis[0].pretty_print()
            break
        elif format_choice == 'p':
            print("(" + paren_str + ")")
            break
        elif format_choice == 'b':
            analysis[0].pretty_print()
            print("(" + paren_str + ")")
            break
        else:
            print("Invalid choice. Please enter 't', 'p', or 'b'.")

if __name__ == "__main__":
    main()



