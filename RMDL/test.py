

import pickle
with open('D:/dic/dicts_from_file.pickle', 'rb') as f:
    dicts_from_file = pickle.load(f)
f.close()

with open('D:/dic/dicts_from_file1.pickle', 'rb') as f:
    dicts_from_file1 = pickle.load(f)
f.close()

with open('D:/dic/dicts_from_file2.pickle', 'rb') as f:
    dicts_from_file2 = pickle.load(f)
f.close()


context = {}
context.update(dicts_from_file1)
context.update(dicts_from_file2)
context.update(dicts_from_file)

def transliterate(line):
    cedilla2latin = [[u'Á', u'A'], [u'á', u'a'], [u'Č', u'C'], [u'č', u'c'], [u'Š', u'S'], [u'š', u's']]
    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line

def translate(text_string):
    for word in text_string.split():
       if word in context:
           if type(context[word]) != list:
            print(str(type(context[word]))+str(len(context[word]))+":"+str(context[word]))
            text_string = text_string.replace(word, str(context[word]))
    return text_string


from html.parser import HTMLParser
import itertools

text = "i luv my iphone yeah, you're awsm apple. Display is Awesome, lol  happppppy "
html_parser = HTMLParser()
text = html_parser.unescape(text)
# text = _slang_loopup(text)
# text = text.decode("utf8").encode('ascii', 'ignore')
# text = " ".join(re.findall('[A-Z][1-9][ ^a-z]*', text))
text = transliterate(text)
text = translate(text)
text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
print(text)