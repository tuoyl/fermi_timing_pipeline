#!/usr/bin/env python
import yaml
import sys

def get_parlist(template):
    with open(template)as fin:
        parlist = yaml.load(fin, Loader=yaml.FullLoader)
    return parlist

def change_configure(write_yamlfile, dictionary, week_num):
    for datakey in dictionary['data']:
        print(datakey, dictionary['data'][datakey])
        dictionary['data'][datakey] = dictionary['data'][datakey].replace("???", str(week_num).zfill(3))
        print(dictionary['data'][datakey])

    with open(write_yamlfile, 'w') as fout:
        documents = yaml.dump(dictionary, fout)


if __name__ == "__main__":
    template = sys.argv[1]
    newyaml  = sys.argv[2]
    week_num = sys.argv[3]

    ## read yaml template
    dictionary = get_parlist(template)

    ## change the key word
    change_configure(newyaml, dictionary, week_num)

