

from transliterate import translit
import pandas as pd

train = 0


if train == 0:
    path = 'spisok'
    model_name = 'wwii'
else:
    path = 'modern_tech'
    model_name = 'modern'
import re

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


labels = []
data_dict = {}
data = []
label = []
labels_to_each = []

with open(path, 'r') as file:
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

        if train == 1:
            if len(labels) != 1 and len(labels) != 8 and len(labels) != 49 and len(labels) != 10:
                sent = line[:-1].split(' ')
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                labels_to_each.append(labels[-1])

                continue
            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            # data.append(line[:-1])
            label.append(len(labels))
            labels_to_each.append(labels[-1])
        else:
            if len(labels) != 2 and len(labels) != 7:
                sent = line[:-1].split(' ')
                sent_new = []
                for t in sent:
                    if has_cyrillic(t) == False:
                        sent_new.append(t)
                sent_st = ' '.join(sent_new)
                if len(sent_st) < 2:
                    continue
                data_dict[sent_st] = len(labels)
                data.append(sent_st)
                label.append(len(labels))
                labels_to_each.append(labels[-1])

                continue
            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            # data.append(line[:-1])
            label.append(len(labels))
            labels_to_each.append(labels[-1])


print(labels_to_each)
df = pd.DataFrame( {'vechile': data,
     'country': labels_to_each,
     'country_number': label
    })
print(df)
df.to_csv('{}.csv'.format(model_name), sep='\t')