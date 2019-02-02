from transliterate import translit
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

labels = []
data_dict = {}
data = []
label = []

with open('spisok', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels.append(line_no_spaces[:-1])
            continue

        for i in range(len(line)):
            if line[i] == ' ':
                continue
            if line[i] == '.':
                line = line[i+2:]
                break

        data_dict[line[:-1]] = len(labels)
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
        label.append(len(labels))
print(labels)
test_str = 'Czołg średni BBT. Br. Panc'
test_str = translit(u"{}".format(test_str), "ru", reversed=True)

data_string = ''
max_len = 0
for i in data:
    data_string += i
    if len(i) > max_len:
        max_len = len(i)

if len(test_str) > max_len:
    test_str = test_str[:max_len]

chars = sorted(list(set(data_string)))
print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))
code_str = [mapping[char] for char in test_str]
print(code_str)
padded = pad_sequences([code_str], maxlen=max_len, padding='post')

model = load_model('models/WWII_names.h5f')


list_of_pred = model.predict(padded)[0]
print(list_of_pred)
top = sorted(range(len(list_of_pred)), key=lambda i: list_of_pred[i], reverse=True)[:5]
for i in top:
    print(labels[i-1], list_of_pred[i])
