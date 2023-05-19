import random
import math
import time
import pandas as pd
import numpy as np
import requests
from collections import defaultdict

import my_dictionaries as d

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.text.splitlines()

path_result_train = 'data\\preprocessing\\GENERATED_raw_logs_train.txt'
path_result_test = 'data\\preprocessing\\GENERATED_raw_logs_test.txt'
path_result_valid ='data\\preprocessing\\GENERATED_raw_logs_valid.txt'

roles_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
commands_probs = []
tables_probs = []
fields_probs = []
# sql_type_dic = {1:"SELECT",2:"INSERT",3:"UPDATE",4:"DELETE"}

tpce_roles_rev = defaultdict(list)
for key, value in d.tpce_roles.items():
    tpce_roles_rev[value].append(int(key))

table_fields_rev = defaultdict(list)
for key, value in d.fields.items():
    table_fields_rev[value[0]].append(key)
# print(tpce_roles_rev)

# getting systemRandom instance out of random class
system_random = random.SystemRandom()
strings_number_values = []


# Secure random number within a range
# print(system_random.randrange(50, 100))
# Output 59

# secure random choice
# list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(system_random.choice(list1))
# Output 9

# secure random sample
# print(system_random.sample(list1, 3))
# Output [1, 4, 9]

# secure random float
# print(system_random.uniform(5.5, 25.5))


# Output 18.6664152449821

def create_select(pr_fields, pr_tables, where, group, order):
    str = f"SELECT {', '.join(pr_fields)} FROM {', '.join(pr_tables)} WHERE {where}"
    if len(group) != 0:
        str += f' GROUP BY {group}'
    if len(order) != 0:
        str += f' ORDER BY {order}'
    return str


# INSERT INTO trade(t_id, t_dts, t_st_id, t_tt_id, t_is_cash, t_s_symb, t_qty, t_bid_price, t_ca_id, t_exec_name, t_trade_price, t_chrg, t_comm, t_tax, t_lifo) VALUES (200000008727428, '2022-12-23 18:14:12', 'PNDG', 'TLB', 1, 'CRYS', 200, 24.91, 43000029033, 'Hubert Bollie', NULL, 3, 15.94, 0, 1)
def create_insert(table, fields, f_values):
    f_values = [str(i) for i in f_values]
    return f"INSERT INTO {table}({', '.join(fields)}) VALUES ({', '.join(f_values)})"


def create_update(table, fields, f_values, where):
    # UPDATE broker SET b_comm_total = b_comm_total + 20.14, b_num_trades = b_num_trades + 1 WHERE b_id = 4300000028
    str = f'UPDATE {table} SET'
    if len(fields) >= 2:
        for i in range(0, len(fields)):
            str += f' {fields[i]} = {f_values[i]},'
        str = str[:-1]
    else:
        str += f'{fields[0]} = {f_values[0]}'
    if len(where) != 0:
        str += f' WHERE {where}'
    return str


def create_delete(table, where):
    return f"DELETE FROM {table} WHERE {where}"


# def generate_random_transaction(r):
#     operations = d.acceptable_operators.get(r)
#     query_types = operations.copy()
#     while len(query_types)<4:
#         query_types.extend(operations)
#
#     rows = tpce_roles_rev.get(r)
#     #used_tables = []
#     frames = []
#     for el in rows:
#         used_tables = []  # ****
#         for tab in d.frames[el]:
#             used_tables.append(tab)
#
#         if len(used_tables)<len(query_types):
#             ex = math.ceil(len(query_types)/len(used_tables))
#             used_tables *=ex
#             used_tables = used_tables[:len(query_types)]
#         else:
#             if len(query_types)< len(used_tables):
#                 ext = math.ceil(len(used_tables) / len(query_types))
#                 query_types *= ext
#                 query_types = query_types[:len(used_tables)]
#                 #print(query_types)
#                 #print('2')
#
#         prep_list = []
#         #print('QU',query_types)
#         for a,b in zip(query_types,used_tables):
#             prep_list.append((a,b))
#
#         field_list = []
#         for el in prep_list:
#             if el[0]==1:
#                 c = system_random.randrange(1,len(table_fields_rev.get(el[1])))
#                 field_list.append(system_random.sample(table_fields_rev.get(el[1]), c))
#             if el[0]==2:
#                 field_list.append(table_fields_rev.get(el[1]))
#             if el[0]==3:
#                 field_list.append(system_random.sample(table_fields_rev.get(el[1]), 1))
#             if el[0]==4:
#                 field_list.append(system_random.sample(table_fields_rev.get(el[1]), 1))
#
#         val_list = []
#         for el in field_list:
#             arr_val = []
#             for e in el:
#                 arr_val.append(system_random.randrange(0,1000000))
#             val_list.append(arr_val)
#         #print(val_list)
#
#         sqls = []
#         for i in range(0,len(prep_list)):
#             if prep_list[i][0]==1:
#                 where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
#                 where_v = system_random.randrange(0, 1000000)
#                 sqls.append(create_select(field_list[i],prep_list[i][1],where_f,where_v))
#             if prep_list[i][0]==2:
#                 sqls.append(create_insert(prep_list[i][1],field_list[i],val_list[i]))
#             if prep_list[i][0]==3:
#                 where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
#                 where_v = system_random.randrange(0, 1000000)
#                 sqls.append(create_update(prep_list[i][1],field_list[i][0],val_list[i][0],where_f,where_v))
#             if prep_list[i][0]==4:
#                 where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
#                 where_v = system_random.randrange(0, 1000000)
#                 sqls.append(create_delete(prep_list[i][1],where_f,where_v))
#         frames.append(sqls)
#     #print(frames)
#     #print(sqls)
#     return frames

# def generate_data(trans_count,roles_list,path_result):
#     ex = math.ceil(trans_count/len(roles_list))
#     r_list =roles_list.copy()
#     r_list *=ex
#     print(len(r_list))
#     #print(len(roles_list))
#     roles = [1]
#     sqls_list =['commit']
#     tr_counter = 0
#     transaction_nums_list = [1]
#     for el in r_list:
#         queries = generate_random_transaction(el)
#         for i in range(0,len(queries)):
#             tr_counter+=1
#             transaction_nums_list.extend([tr_counter]*(len(queries[i])+1))
#             sqls_list.extend(queries[i])
#             sqls_list.append('commit')
#             roles.extend([el]*(len(queries[i])+1))
#
#
#     data = pd.DataFrame(list(zip(transaction_nums_list,roles,sqls_list)),columns=['transaction','role','query'])
#     data.to_csv(path_result, sep='\t',index=False, header=False)
#     #data.to_csv('after.txt', sep='\t', index=False, header=False)
#     print('DONE')
def make_special_tokens():
    list_keys = list(d.fields.keys())
    v = list(d.fields.values())
    vals = [el[0] for el in v]
    list_values = list(set(vals))
    print(list_values)
    print(list_keys)
    result_tokens = list_values + list_keys
    f = open('tokens.txt', 'w')
    for t in result_tokens:
        f.write("%s\n" % t)


def zipf(x, X):
    return (x ** (-0.5)) / sum([i ** -0.5 for i in X])


def calc_probs():
    for i in range(1, 5):
        commands_probs.append(zipf(i, [1, 2, 3, 4]))
    for j in range(1, len(table_fields_rev) + 1):
        tables_probs.append(zipf(j, list(range(1, len(table_fields_rev) + 1))))
    for t in table_fields_rev.keys():
        f_probs = []
        for f in range(1, len(table_fields_rev.get(t)) + 1):
            f_probs.append(zipf(f, list(range(1, len(table_fields_rev.get(t)) + 1))))
        fields_probs.append(f_probs)
    # for k in range(1,len):


def prob_C(r):
    com = []
    for c in d.acceptable_operators.get(r):
        com.append(zipf(c, list(range(1, len(d.acceptable_operators.get(r)) + 1))) / 1)
    return com  # Возвращает массив вероятностей разрешенных операторов


def prob_Pt(f):
    tabs = d.frames[f]
    weights = []
    for t in range(1, len(tabs) + 1):
        weights.append(zipf(t, list(range(1, len(tabs) + 1))))
    return weights  # Возвращает массив вероятностей разрешенных таблиц. роль -> фрейм-> разрешенные таблицы


def prob_St(selected_tables):
    weights = []
    for t in range(1, len(selected_tables) + 1):
        weights.append(zipf(t, list(range(1, len(selected_tables) + 1))))
    return weights  # Для отобранных разрешенных таблиц возвращает вероятности


def prob_Pa(select_tables):
    weights = []
    for t in select_tables:
        for f in range(1, (len(table_fields_rev.get(t)) + 1)):
            weights.append(zipf(f, list(range(1, len(table_fields_rev.get(t)) + 1))))
    return weights


def prob_Sa(where_tables):
    weights = []
    for t in where_tables:
        for f in range(1, len(table_fields_rev.get(t)) + 1):
            weights.append(zipf(f, list(range(1, len(table_fields_rev.get(t)) + 1))))
    return weights


def create_val(bool_type):
    if not bool_type:
        return system_random.randrange(1, 101)
    else:
        return f'\'{system_random.choice(WORDS)}\''


def prob_AO(where_fields, c, r):
    f_types = [d.fields.get(x)[2] for x in where_fields]
    str = ""
    AND = False
    OR = False
    if len(where_fields) >= 2:
        AND = True
        OR = True
    if c == 1:
        if r in [5, 8, 10]:
            if OR:
                str += f'{where_fields[0]} = {create_val(f_types[0])} OR {where_fields[1]} = {create_val(f_types[1])}'
            else:
                str += f'{where_fields[0]} = {create_val(f_types[0])}'
        else:
            if AND:
                str += f'{where_fields[0]} = {create_val(f_types[0])} AND {where_fields[1]} = {create_val(f_types[1])}'
            else:
                str += f'{where_fields[0]} = {create_val(f_types[0])}'

    if c == 3:
        if AND:
            str += f'{where_fields[0]} = {create_val(f_types[0])}'
            for i in range(1, len(where_fields)):
                str += f' AND {where_fields[i]} = {create_val(f_types[i])}'
        else:
            str += f'{where_fields[0]} = {create_val(f_types[0])}'
    if c == 4:
        if AND:
            str += f'{where_fields[0]} = {create_val(f_types[0])} AND {where_fields[1]} = {create_val(f_types[1])}'
        else:
            str += f'{where_fields[0]} = {create_val(f_types[0])}'
    return str


def prob_GroupBy(proj_fields, c, r):
    str = ""
    if c != 1:
        return str
    else:
        if (r in [2, 6, 9, 10]) and (len(proj_fields) >= 2):
            str += f'{proj_fields[0]}'
    return str


def prob_OrderBy(proj_fields, c, r):
    str = ""
    if c != 1:
        return str
    else:
        if (len(proj_fields) % 2 == 0) and (len(proj_fields) >= 2) and r in [1, 2, 3, 4, 7, 10, 11]:
            str += f'{proj_fields[0]}'
    return str


def make_query(c, r, proj_fields, select_tables, where_fields, groupby_part, orderby_part):
    query = ''
    if c == 1:
        where_part = prob_AO(where_fields, c, r)
        query = create_select(proj_fields, select_tables, where_part, groupby_part, orderby_part)
    if c == 2:
        table = select_tables[0]
        f_types = [d.fields.get(x)[2] for x in table_fields_rev.get(select_tables[0])]
        vals = [create_val(x) for x in f_types]
        query = create_insert(select_tables[0], table_fields_rev.get(select_tables[0]), vals)
    if c == 3:
        table = d.fields.get(where_fields[0])[0]
        fields = list(set(where_fields) & set(table_fields_rev.get(table)))
        fields.sort()
        where_part = prob_AO(fields, c, r)
        f_types = [d.fields.get(x)[2] for x in fields]
        vals = [create_val(x) for x in f_types]
        query = create_update(table, fields, vals, where_part)
    if c == 4:
        # fields = list(set(proj_fields) & set(table_fields_rev.get(select_tables[0])))
        # fields.sort()
        # Нужно выбрать одну таблицу из всех where fields и с ней уже работать делитом
        table = d.fields.get(where_fields[0])[0]
        fields = list(set(where_fields) & set(table_fields_rev.get(table)))
        where_part = prob_AO(fields, c, r)
        query = create_delete(table, where_part)
    #print(query)
    return query


def gen(count_for_each_role,path):
    iterate_r_list = roles_list.copy()
    iterate_r_list *= count_for_each_role
    tr_list = []
    r_list = []
    q_list = []
    tr_counter = 0
    for r in iterate_r_list:
        tr_counter += 1
        tr_len = system_random.randrange(d.transaction_length.get(r)[0], d.transaction_length.get(r)[1] + 1)
        for j in range(0, tr_len):
            l1 = []
            l2 = []
            # Формирование ОДНОГО SQL-ЗАПРОСА
            c = system_random.choices(d.acceptable_operators.get(r), weights=prob_C(r), k=1)[0]
            # Случайный фрейм из доступных
            f = system_random.choice(tpce_roles_rev.get(r))
            Pt_count = system_random.choice(list(range(1, len(d.frames[f]) + 1)))
            select_tables = system_random.choices(d.frames[f], weights=prob_Pt(f),k=Pt_count)
            select_tables = list(set(select_tables))
            Pt_count = len(select_tables)
            where_tab_count = system_random.choice(list(range(1, len(select_tables) + 1)))
            where_tables = system_random.choices(select_tables, weights=prob_St(select_tables), k=where_tab_count)
            where_tables=list(set(where_tables))
            where_tab_count = len(where_tables)
            proj_f_count = system_random.choice(list(range(1, sum([len(table_fields_rev.get(x)) for x in select_tables]) + 1)))
            [l1.extend(table_fields_rev.get(x)) for x in select_tables]
            proj_fields = system_random.choices(l1, weights=prob_Pa(select_tables), k=proj_f_count)
            proj_fields = list(set(proj_fields))
            proj_f_count=len(proj_fields)
            where_f_count = system_random.choice(list(range(1, sum([len(table_fields_rev.get(x)) for x in where_tables]) + 1)))
            [l2.extend(table_fields_rev.get(x)) for x in where_tables]
            where_fields = system_random.choices(l2, weights=prob_Sa(where_tables), k=where_f_count)
            where_fields=list(set(where_fields))
            where_f_count=len(where_fields)
            groupby_part = prob_GroupBy(proj_fields, c, r)
            orderby_part = prob_OrderBy(proj_fields, c, r)
            query = make_query(c, r, proj_fields, select_tables, where_fields, groupby_part, orderby_part)

            tr_list.append(tr_counter)
            r_list.append(r)
            q_list.append(query)

        tr_list.append(tr_counter)
        r_list.append(r)
        q_list.append('commit')

    data = pd.DataFrame(list(zip(tr_list, r_list, q_list)), columns=['transaction', 'role', 'query'])
    data.to_csv(path, sep='\t', index=False, header=False)
    print(f'Done {path}')


gen(1000,path_result_train)
gen(100,path_result_valid)
gen(100,path_result_test)

