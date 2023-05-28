import pandas as pd
import json
import re
import sqlparse
from sqlparse.sql import IdentifierList, TokenList
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import my_dictionaries as d
MAX_COLUMNS = 24
NUM_TABLES = 33
query_mode = {'SELECT': [1, 6, 2], 'INSERT': [2, 4, 4], 'UPDATE': [3, 2, 6], 'DELETE': [4, 4]}
source_data_train = 'data\\preprocessing\\GENERATED_raw_logs_train.txt'
source_data_test = 'data\\preprocessing\\GENERATED_raw_logs_test.txt'

result_data_train_vec = 'data\\train\\GENERATED_new_vectors_train.json'
result_data_test_vec = 'data\\test\\GENERATED_new_vectors_test.json'

tpce_roles_rev = defaultdict(list)
for key, value in d.tpce_roles.items():
    tpce_roles_rev[value].append(int(key))

table_fields_rev = defaultdict(list)
for key, value in d.fields.items():
    table_fields_rev[value[0]].append(key)
# print(tpce_roles_rev)



def centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = [sum(x) / len(points), sum(y) / len(points)]
    return centroid

def get_field_point(field):
    p =[0.0,0.0]
    p[0] =  d.tables.get(d.fields.get(field)[0])
    p[1] = d.fields.get(field)[1]+1
    return p

def get_table_point(table,z):
    p = [0,z]
    p[0] = d.tables.get(str(table))
    return p

def SQL_CMD(query):
    sql_cmd = [0.0, 0.0]
    parsed = sqlparse.parse(query)[0]
    type = parsed.tokens[0].value
    sql_cmd[0] = query_mode[str(type)][0]
    sql_cmd[1] = len(query)
    return sql_cmd

def PROJ_REL(query):
    query = query.replace(" FORCE INDEX(PRIMARY)", "")
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    proj_rel = [0, 0]
    t_list = []
    parsed = sqlparse.parse(query)[0]
    t = parsed.tokens[0].value
    idx = 0
    for i in range(0, len(parsed.tokens)):
        if (str(parsed.tokens[i].value) == 'FROM'):
            idx = i + 2

    if idx != 0:
        a = parsed.tokens[idx]
        b = TokenList([a])
        c = IdentifierList(b)
        f = c.get_identifiers()
        e = list(f)
        proj_rel[0] = len(e)
        tokenizer = RegexpTokenizer(r'\w+')
        S = re.compile(r'FROM([^"]*)WHERE')
        t = S.findall(query)
        tokens = tokenizer.tokenize(str(t))
        for elem in tokens:
            if d.tables.get(str(elem)):
                t_list.append(elem)
    return t_list

def PROJ_ATTR(query):
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    f_list = []
    new_list =[]
    t = parsed.tokens[0].value
    if t == 'SELECT':
        tokenizer = RegexpTokenizer(r'\w+')
        S = re.compile(r'SELECT([^"]*)FROM')
        t = S.findall(query)
        ilist = tokenizer.tokenize(str(t))
        for elem in ilist:
            if d.fields.get(str(elem)):
                new_list.append(elem)
        f_list = list(set(new_list))
    return f_list

def SEL_ATTR_REL(query):
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    idx = 0
    f_list = []
    t_list = []
    for i in range(0, len(parsed.tokens)):
        if (str(type(parsed.tokens[i])) == '<class \'sqlparse.sql.Where\'>'):
            idx = i

    if idx != 0:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        for elem in tokens:
            if d.fields.get(str(elem)):
                f_list.append(elem)
        for el in f_list:
            t_list.append(d.fields[el][0])

    if parsed.tokens[0].value == 'INSERT':
        tokenizer = RegexpTokenizer(r'\w+')
        S = re.compile(r'INTO([^"]*)VALUES')
        t = S.findall(query)
        ilist = tokenizer.tokenize(str(t))
        for elem in ilist:
            if d.fields.get(str(elem)):
                f_list.append(elem)
            if d.tables.get(str(elem)):
                t_list.append(elem)

    f_list = list(set(f_list))
    t_list = list(set(t_list))
    return t_list,f_list

def VALUE_CTR(query):
    value_ctr = [0.0, 0.0]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    copy_query = query
    strings = re.compile(r'[\'\"]([^\'\"]+)[\'\"][,\s)]?')
    strings_list = strings.findall(copy_query)
    if (len(strings_list)) > 0:
        value_ctr[0] = len(strings_list)
        s = "".join(strings_list)
        for el in strings_list:
            copy_query = copy_query.replace(str(el), "xxx")

    copy_query = copy_query.replace("LIMIT ", "xxx")

    numeric = re.compile(r'[\s(]([\d.]+)[,\s)]?')
    numeric_list = numeric.findall(copy_query)
    if len(numeric_list) > 0:
        value_ctr[1] = len(numeric_list)


    return value_ctr

def make_vector(query):
    vector = []
    first_vector = SQL_CMD(query)
    proj_t = PROJ_REL(query)
    proj_t.sort()
    proj_f = PROJ_ATTR(query)
    sel_t,sel_f = SEL_ATTR_REL(query)
    sel_t.sort()
    vals = VALUE_CTR(query)
    vector.extend(first_vector)
    #0 - нихрена 1 - relation 2 -  selection
    relation_t_block = [[0.0,0.0] for i in range(0,NUM_TABLES)]
    for el in proj_t:
        relation_t_block[d.tables.get(el)]= get_table_point(el,1)
    relation_f_block = [[0.0,0.0] for i in range(0,NUM_TABLES)]
    for t in proj_t:
        l = list(set(proj_f)& set(table_fields_rev.get(t)))
        l2 = [get_field_point(x) for x in l]
        if len(l2)>0:
            c =  centroid(l2)
            relation_f_block[d.tables.get(t)] =c
    selection_t_block = [[0.0,0.0] for i in range(0,NUM_TABLES)]
    for el in sel_t:
        selection_t_block[d.tables.get(el)]=get_table_point(el,2)
    selection_f_block= [[0.0,0.0] for i in range(0,NUM_TABLES)]
    for t in sel_t:
        l = list(set(sel_f) & set(table_fields_rev.get(t)))
        l2 = [get_field_point(x) for x in l]
        if len(l2)>0:
            c = centroid(l2)
            selection_f_block[d.tables.get(t)] = c
    resulted = first_vector
    resulted.extend(float(item) for sublist in relation_t_block for item in sublist)
    resulted.extend(float(item) for sublist in relation_f_block for item in sublist)
    resulted.extend(float(item) for sublist in selection_t_block for item in sublist)
    resulted.extend(float(item) for sublist in selection_f_block for item in sublist)
    resulted.extend(float(v) for v in vals)
    return resulted
def make_transaction_new_vectors(source, result):
    #
    data = pd.read_csv(source, sep="\t", header=None)
    data.columns = ["transaction", 'role', "query"]
    data.drop(data[data['query'] == 'commit'].index, inplace=True)
    data.reset_index()
    new_list =[]
    role_list_new =[]

    data['query'] = data['query'].apply(lambda x: make_vector(x))
    for i in range (1,data.iloc[-1]['transaction']+1):
        trans_num=i
        new_vector = []
        role = 0
        queries = data.loc[data['transaction'] == i]
        for index, rows in queries.iterrows():
            new_vector= new_vector+list(rows['query'])
            role = rows['role']
        ext= 5360 - len(new_vector)
        new_vector = new_vector+[0.0]*ext
        new_list.append(new_vector)
        role_list_new.append(role)

    data = pd.DataFrame(list(zip(role_list_new,new_list)),columns =['role','query'])

    d = data.to_dict('records')
    f = open(result, 'w', encoding='utf-8')
    json.dump(d, f, ensure_ascii=False)
    print(f'Done {result}')

#268
#max 20
#5360

make_transaction_new_vectors(source_data_train,result_data_train_vec)
make_transaction_new_vectors(source_data_test,result_data_test_vec)