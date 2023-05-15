import random
import math
import time
import pandas as pd
import numpy as np
from collections import defaultdict

path_result = 'data\\preprocessing\\GENERATED_raw_logs_1.txt'
path_result_test = 'data\\preprocessing\\GENERATED_raw_logs_test_1.txt'

import my_dictionaries as d
roles_list = [1,2,3,4,5,6,7,8,9,10,11]
# sql_type_dic = {1:"SELECT",2:"INSERT",3:"UPDATE",4:"DELETE"}

tpce_roles_rev = defaultdict(list)
for key, value in d.tpce_roles.items():
    tpce_roles_rev[value].append(int(key))

table_fields_rev = defaultdict(list)
for key, value in d.fields.items():
    table_fields_rev[value[0]].append(key)
#print(tpce_roles_rev)
#print(table_fields_rev)

# getting systemRandom instance out of random class
system_random = random.SystemRandom()
# Secure random number
#print(system_random.randint(1, 30))
# Output 22

# Secure random number within a range
#print(system_random.randrange(50, 100))
# Output 59

# secure random choice
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#print(system_random.choice(list1))
# Output 9

# secure random sample
#print(system_random.sample(list1, 3))
# Output [1, 4, 9]

# secure random float
#print(system_random.uniform(5.5, 25.5))


# Output 18.6664152449821

def create_select(fields, tables, where_f, where_val):
    return f"SELECT {', '.join(fields)} FROM {tables} WHERE {where_f} = {where_val}"


# INSERT INTO trade(t_id, t_dts, t_st_id, t_tt_id, t_is_cash, t_s_symb, t_qty, t_bid_price, t_ca_id, t_exec_name, t_trade_price, t_chrg, t_comm, t_tax, t_lifo) VALUES (200000008727428, '2022-12-23 18:14:12', 'PNDG', 'TLB', 1, 'CRYS', 200, 24.91, 43000029033, 'Hubert Bollie', NULL, 3, 15.94, 0, 1)
def create_insert(table, fields, f_values):
    f_values = [str(i) for i in f_values]
    return f"INSERT INTO {table}({', '.join(fields)}) VALUES ({', '.join(f_values)})"


def create_update(table, field, field_value, where_f, where_val):
    # UPDATE broker SET b_comm_total = b_comm_total + 20.14, b_num_trades = b_num_trades + 1 WHERE b_id = 4300000028
    return f"UPDATE {table} SET {field} = {field_value} WHERE {where_f} = {where_val}"


def create_delete(table, where_f, where_val):
    return f"DELETE FROM {table} WHERE {where_f} = {where_val}"



def generate_random_transaction(r):
    operations = d.acceptable_operators.get(r)
    query_types = operations.copy()
    while len(query_types)<4:
        query_types.extend(operations)
    #query_types = operations + system_random.sample(operations, (query_amount - len(operations)))
    #print(query_types)

    rows = tpce_roles_rev.get(r)
    used_tables = []
    for el in rows:
        for tab in d.frames[el]:
            used_tables.append(tab)

    #print(used_tables)
    if len(used_tables)<len(query_types):
        ex = math.ceil(len(query_types)/len(used_tables))
        used_tables *=ex
        used_tables = used_tables[:len(query_types)]
        #print(used_tables)
        #print('1')
    else:
        if len(query_types)< len(used_tables):
            ext = math.ceil(len(used_tables) / len(query_types))
            query_types *= ext
            query_types = query_types[:len(used_tables)]
            #print(query_types)
            #print('2')

    prep_list = []
    #print('QU',query_types)
    for a,b in zip(query_types,used_tables):
        prep_list.append((a,b))

    #print('PREP', len(prep_list), prep_list)
    #print(prep_list)
    field_list = []
    for el in prep_list:
        if el[0]==1:
            c = system_random.randrange(1,len(table_fields_rev.get(el[1])))
            field_list.append(system_random.sample(table_fields_rev.get(el[1]), c))
        if el[0]==2:
            field_list.append(table_fields_rev.get(el[1]))
        if el[0]==3:
            field_list.append(system_random.sample(table_fields_rev.get(el[1]), 1))
        if el[0]==4:
            field_list.append(system_random.sample(table_fields_rev.get(el[1]), 1))
    #print(field_list)

    val_list = []
    for el in field_list:
        arr_val = []
        for e in el:
            arr_val.append(system_random.randrange(0,1000000))
        val_list.append(arr_val)
    #print(val_list)

    sqls = []
    for i in range(0,len(prep_list)):
        if prep_list[i][0]==1:
            where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
            where_v = system_random.randrange(0, 1000000)
            sqls.append(create_select(field_list[i],prep_list[i][1],where_f,where_v))
        if prep_list[i][0]==2:
            sqls.append(create_insert(prep_list[i][1],field_list[i],val_list[i]))
        if prep_list[i][0]==3:
            where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
            where_v = system_random.randrange(0, 1000000)
            sqls.append(create_update(prep_list[i][1],field_list[i][0],val_list[i][0],where_f,where_v))
        if prep_list[i][0]==4:
            where_f = system_random.choice(table_fields_rev.get(prep_list[i][1]))
            where_v = system_random.randrange(0, 1000000)
            sqls.append(create_delete(prep_list[i][1],where_f,where_v))
    #print(sqls)
    return sqls

def generate_data(trans_count,roles_list,path_result):
    ex = math.ceil(trans_count/len(roles_list))
    r_list =roles_list.copy()
    r_list *=ex
    print(len(r_list))
    #print(len(roles_list))
    roles = [1]
    sqls_list =['commit']
    tr_counter = 0
    transaction_nums_list = [1]
    for el in r_list:
        tr_counter +=1
        queries = generate_random_transaction(el)
        transaction_nums_list.extend([tr_counter]*(len(queries)+1))
        sqls_list.extend(queries)
        sqls_list.append('commit')
        roles.extend([el]*(len(queries)+1))

    # print(len(roles))
    # print(len(sqls_list))
    # print(len(transaction_nums_list))
    data = pd.DataFrame(list(zip(transaction_nums_list,roles,sqls_list)),columns=['transaction','role','query'])
    #print(data)
    data.to_csv(path_result, sep='\t',index=False, header=False)
    print('DONE')

generate_data(11*1000,roles_list,path_result)
generate_data(11*100,roles_list,path_result_test)