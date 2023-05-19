import random
from collections import Counter

import numpy as np
import pandas as pd
import re
import time
import regex
import json
import sqlparse
import swifter
from sklearn.cluster import KMeans
from sqlparse.sql import IdentifierList, TokenList, Parenthesis
from nltk.tokenize import RegexpTokenizer


MAX_COLUMNS = 24
NUM_TABLES = 33

source_data_train = 'data\\preprocessing\\GENERATED_raw_logs_train.txt'
source_data_test = 'data\\preprocessing\\GENERATED_raw_logs_test.txt'
source_data_valid = 'data\\preprocessing\\GENERATED_raw_logs_valid.txt'

result_data_train_classic = 'data\\train\\GENERATED_classic_vectors_train.json'
result_data_test_classic = 'data\\test\\GENERATED_classic_vectors_test.json'
result_data_valid_classic = 'data\\train\\GENERATED_classic_vectors_valid.json'


result_data_train_transaction = 'data\\train\\GENERATED_transaction_vectors_train.json'
result_data_test_transaction = 'data\\test\\GENERATED_transaction_vectors_test.json'
result_data_valid_transaction= 'data\\train\\GENERATED_transaction_vectors_valid.json'


lead_time = 'data\\making_vectors.json'

# path_trainXvector1 = 'data\\train\\train_vectors31.csv'
# path_testXvector1 = 'data\\test\\test_vectors41.csv'
# path_testXvectorAnomal1 = 'data\\test\\test_vectors_anomaly42.csv'


type_matrix = np.array(
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 2, 3, 4], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 4],
     [1, 0, 0, 0], [1, 0, 3, 0], [1, 0, 3, 0]])
tables = {'account_permission': 0, 'address': 1, 'broker': 2, 'cash_transaction': 3, 'charge': 4, 'commission_rate': 5,
          'company': 6, 'company_competitor': 7, 'customer': 8,
          'customer_account': 9, 'customer_taxrate': 10, 'daily_market': 11, 'exchange': 12, 'financial': 13,
          'holding': 14, 'holding_history': 15, 'holding_summary': 16,
          'industry': 17, 'last_trade': 18, 'news_xref': 19, 'sector': 20, 'security': 21,
          'seq_trade_id': 22, 'settlement': 23, 'status_type': 24, 'taxrate': 25,
          'trade': 26, 'trade_history': 27, 'trade_request': 28, 'trade_type': 29, 'watch_item': 30, 'watch_list': 31,
          'zip_code': 32}

# [0]- тип запроса [1] - индекс листа с таблицами [2] - индекс листа с полями
query_mode = {'SELECT': [1, 6, 2], 'INSERT': [2, 4, 4], 'UPDATE': [3, 2, 6], 'DELETE': [4, 4]}
tabu_list = ['SELECT LAST_INSERT_ID()', 'SET TRANSACTION ISOLATION LEVEL READ COMMITTED',
             'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ', 'SET TRANSACTION ISOLATION LEVEL SERIALIZABLE']
roles = {}

fields = {'ap_ca_id': ['account_permission', 0], 'ap_acl': ['account_permission', 1],
          'ap_tax_id': ['account_permission', 2],
          'ap_l_name': ['account_permission', 3], 'ap_f_name': ['account_permission', 4], 'ad_id': ['address', 0],
          'ad_line1': ['address', 1],
          'ad_line2': ['address', 2], 'ad_zc_code': ['address', 3], 'ad_ctry': ['address', 4], 'b_id': ['broker', 0],
          'b_st_id': ['broker', 1],
          'b_name': ['broker', 2], 'b_num_trades': ['broker', 3], 'b_comm_total': ['broker', 4],
          'ct_t_id': ['cash_transaction', 0],
          'ct_dts': ['cash_transaction', 1], 'ct_amt': ['cash_transaction', 2], 'ct_name': ['cash_transaction', 3],
          'ch_tt_id': ['charge', 0],
          'ch_c_tier': ['charge', 1], 'ch_chrg': ['charge', 2], 'cr_c_tier': ['commission_rate', 0],
          'cr_tt_id': ['commission_rate', 1],
          'cr_ex_id': ['commission_rate', 2], 'cr_from_qty': ['commission_rate', 3],
          'cr_to_qty': ['commission_rate', 4], 'cr_rate': ['commission_rate', 5],
          'co_id': ['company', 0], 'co_st_id': ['company', 1], 'co_name': ['company', 2], 'co_in_id': ['company', 3],
          'co_sp_rate': ['company', 4],
          'co_ceo': ['company', 5], 'co_ad_id': ['company', 6], 'co_desc': ['company', 7],
          'co_open_date': ['company', 8],
          'cp_co_id': ['company_competitor', 0], 'cp_comp_co_id': ['company_competitor', 1],
          'cp_in_id': ['company_competitor', 2],
          'c_id': ['customer', 0], 'c_tax_id': ['customer', 1], 'c_st_id': ['customer', 2], 'c_l_name': ['customer', 3],
          'c_f_name': ['customer', 4],
          'c_m_name': ['customer', 5], 'c_gndr': ['customer', 6], 'c_tier': ['customer', 7], 'c_dob': ['customer', 8],
          'c_ad_id': ['customer', 9],
          'c_ctry_1': ['customer', 10], 'c_area_1': ['customer', 11], 'c_local_1': ['customer', 12],
          'c_ext_1': ['customer', 13], 'c_ctry_2': ['customer', 14],
          'c_area_2': ['customer', 15], 'c_local_2': ['customer', 16], 'c_ext_2': ['customer', 17],
          'c_ctry_3': ['customer', 18], 'c_area_3': ['customer', 19],
          'c_local_3': ['customer', 20], 'c_ext_3': ['customer', 21], 'c_email_1': ['customer', 22],
          'c_email_2': ['customer', 23], 'ca_id': ['customer_account', 0],
          'ca_b_id': ['customer_account', 1], 'ca_c_id': ['customer_account', 2], 'ca_name': ['customer_account', 3],
          'ca_tax_st': ['customer_account', 4],
          'ca_bal': ['customer_account', 5], 'cx_tx_id': ['customer_taxrate', 0], 'cx_c_id': ['customer_taxrate', 1],
          'dm_date': ['daily_market', 0],
          'dm_s_symb': ['daily_market', 1], 'dm_close': ['daily_market', 2], 'dm_high': ['daily_market', 3],
          'dm_low': ['daily_market', 4], 'dm_vol': ['daily_market', 5],
          'ex_id': ['exchange', 0], 'ex_name': ['exchange', 1], 'ex_num_symb': ['exchange', 2],
          'ex_open': ['exchange', 3], 'ex_close': ['exchange', 4],
          'ex_desc': ['exchange', 5], 'ex_ad_id': ['exchange', 6], 'fi_co_id': ['financial', 0],
          'fi_year': ['financial', 1], 'fi_qtr': ['financial', 2],
          'fi_qtr_start_date': ['financial', 3], 'fi_revenue': ['financial', 4], 'fi_net_earn': ['financial', 5],
          'fi_basic_eps': ['financial', 6],
          'fi_dilut_eps': ['financial', 7], 'fi_margin': ['financial', 8], 'fi_inventory': ['financial', 9],
          'fi_assets': ['financial', 10],
          'fi_liability': ['financial', 11], 'fi_out_basic': ['financial', 12], 'fi_out_dilut': ['financial', 13],
          'h_t_id': ['holding', 0],
          'h_ca_id': ['holding', 1], 'h_s_symb': ['holding', 2], 'h_dts': ['holding', 3], 'h_price': ['holding', 4],
          'h_qty': ['holding', 5],
          'hh_h_t_id': ['holding_history', 0], 'hh_t_id': ['holding_history', 1],
          'hh_before_qty': ['holding_history', 2], 'hh_after_qty': ['holding_history', 3],
          'hs_ca_id': ['holding_summary', 0], 'hs_s_symb': ['holding_summary', 1], 'hs_qty': ['holding_summary', 2],
          'in_id': ['industry', 0],
          'in_name': ['industry', 1], 'in_sc_id': ['industry', 2], 'lt_s_symb': ['last_trade', 0],
          'lt_dts': ['last_trade', 1], 'lt_price': ['last_trade', 2],
          'lt_open_price': ['last_trade', 3], 'lt_vol': ['last_trade', 4], 'nx_ni_id': ['news_xref', 0],
          'nx_co_id': ['news_xref', 1],
          'sc_id': ['sector', 0], 'sc_name': ['sector', 1], 's_symb': ['security', 0], 's_issue': ['security', 1],
          's_st_id': ['security', 2],
          's_name': ['security', 3], 's_ex_id': ['security', 4], 's_co_id': ['security', 5],
          's_num_out': ['security', 6], 's_start_date': ['security', 7],
          's_exch_date': ['security', 8], 's_pe': ['security', 9], 's_52wk_high': ['security', 10],
          's_52wk_high_date': ['security', 11],
          's_52wk_low': ['security', 12], 's_52wk_low_date': ['security', 13], 's_dividend': ['security', 14],
          's_yield': ['security', 15],
          'id': ['seq_trade_id', 0], 'se_t_id': ['settlement', 0], 'se_cash_type': ['settlement', 1],
          'se_cash_due_date': ['settlement', 2],
          'se_amt': ['settlement', 3], 'st_id': ['status_type', 0], 'st_name': ['status_type', 1],
          'tx_id': ['taxrate', 0], 'tx_name': ['taxrate', 1],
          'tx_rate': ['taxrate', 2], 't_id': ['trade', 0], 't_dts': ['trade', 1], 't_st_id': ['trade', 2],
          't_tt_id': ['trade', 3], 't_is_cash': ['trade', 4],
          't_s_symb': ['trade', 5], 't_qty': ['trade', 6], 't_bid_price': ['trade', 7], 't_ca_id': ['trade', 8],
          't_exec_name': ['trade', 9],
          't_trade_price': ['trade', 10], 't_chrg': ['trade', 11], 't_comm': ['trade', 12], 't_tax': ['trade', 13],
          't_lifo': ['trade', 14],
          'th_t_id': ['trade_history', 0], 'th_dts': ['trade_history', 1], 'th_st_id': ['trade_history', 2],
          'tr_t_id': ['trade_request', 0],
          'tr_tt_id': ['trade_request', 1], 'tr_s_symb': ['trade_request', 2], 'tr_qty': ['trade_request', 3],
          'tr_bid_price': ['trade_request', 4],
          'tr_b_id': ['trade_request', 5], 'tt_id': ['trade_type', 0], 'tt_name': ['trade_type', 1],
          'tt_is_sell': ['trade_type', 2], 'tt_is_mrkt': ['trade_type', 3],
          'wi_wl_id': ['watch_item', 0], 'wi_s_symb': ['watch_item', 1], 'wl_id': ['watch_list', 0],
          'wl_c_id': ['watch_item', 1],
          'zc_code': ['zip_code', 0], 'zc_town': ['zip_code', 1], 'zc_div': ['zip_code', 2]}
# 1BV - 0, 2CP-1,2 , 3MF - 3, 4MW -4,5SD-5, 6TL -678 , 7TO -9,10,11,12 ,8TR - 13,14,15,16,17,18 , 9TS - 19, 10TU - 20,21,22 , 11DM - 23 ,12TC - 24
frames = [['broker', 'trade_request'],
          ['customer', 'customer_account', 'holding summary', 'last_trade'], ['status_type', 'trade_history', 'trade'],
          ['last_trade', 'trade', 'trade_history', 'trade_request'],
          ['company', 'daily_market', 'holding_summary', 'industry', 'last_trade', 'security', 'watch_item',
           'watch_list'],
          ['address', 'company', 'company_competitor', 'daily_market', 'exchange', 'financial', 'industry',
           'last_trade', 'news_item', 'news_xref', 'security', 'zip_code'],
          ['cash_transaction', 'settlement', 'trade', 'trade_history', 'trade_type'],
          ['cash_transaction', 'settlement', 'trade', 'trade_history'], ['holding_history', 'trade'],
          ['broker', 'customer', 'customer_account'], ['account_permission'],
          ['charge', 'comission_rate', 'company', 'customer_account', 'customer_taxrate', 'holding', 'holding_summary',
           'last_trade', 'security', 'taxrate', 'trade_type'], ['trade', 'trade_history', 'trade_request'],
          ['holding_summary', 'trade', 'trade_type'],
          ['customer_account', 'holding', 'holding_summary', 'holding_history'],
          ['customer_taxrate', 'taxrate', 'trade'], ['comission_rate', 'customer', 'security'],
          ['broker', 'trade', 'trade_history'], ['cash_transaction', 'customer_account', 'settlement'],
          ['broker', 'customer', 'exchange', 'security', 'status_type', 'trade', 'trade_type'],
          ['cash_transaction', 'settlement', 'trade', 'trade_history', 'trade_type'],
          ['cash_transaction', 'settlement', 'trade', 'trade_history'],
          ['cash_transaction', 'security', 'settlement', 'trade', 'trade_history', 'trade_type'],
          ['account_permission', 'address', 'company', 'customer', 'customer_taxrate', 'daily_market', 'exchange',
           'financial', 'security', 'news_item', 'news_xref', 'taxrate', 'watch_item', 'watch_list'],
          ['trade', 'trade_history', 'trade_request']]
tpce_roles = {'0': 1, '1': 2, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 6, '8': 6, '9': 7, '10': 7, '11': 7, '12': 7,
              '13': 8, '14': 8, '15': 8, '16': 8, '17': 8, '18': 8, '19': 9, '20': 10, '21': 10, '22': 10, '23': 11,
              '24': 12}


def transaction_numbering(data):
    num = 0
    print('drop waste')
    print(data.last_valid_index())
    for index, rows in data.iterrows():
        print(index)
        if data.loc[index]['query'] == 'rollback':
            data.loc[index]["query"] = 'commit'
        if data.loc[index]['query'] in tabu_list:
            data.drop([index], inplace=True)
    print('numbering')
    # count = data.shape[0]
    # print('count',count)
    # data.reindex(range(count), method='ffill')
    last = data.last_valid_index()
    print("last",last)
    for index, rows in data.iterrows():
        if index != last:
            print(index)
            # print(rows)
            if data.loc[index]['query'] == 'commit':
                #if data.loc[int(index) + 1]['query'] != 'commit':
                num = num + 1
        # rows['transaction'] = num
        data.loc[index]['transaction'] = num
    print("numbering ends")


def transaction_iterator(data, path_result):
    data.columns = ["transaction", 'role', "query"]
    count = int(data['transaction'].max()) #+ 1
    count = 10000
    combo_list = []
    type_list = []
    new_data_transactions = []
    new_data_vectors = []
    for i in range(1, count + 1):
        trans_num = i
        # print('Транзакция ',trans_num)
        combo_list.clear()
        type_list.clear()
        queries = data.loc[data['transaction'] == i]
        for index, rows in queries.iterrows():
            if rows['query'] != "commit":
                #print(rows['query'])
                vector, t_list = classic_vector_extractor(rows['query'])
                combo_list = combo_list + t_list
                type_list.append(vector[0])
                new_data_vectors.append(vector)
                new_data_transactions.append(trans_num)

        role_determinant(combo_list, type_list, trans_num)
    #roles_file = open('data\\preprocessing\\roles.txt')
    #json.dump(roles,roles_file,ensure_ascii=False)
    roles_l = []
    for el in (new_data_transactions):
        roles_l.append(roles.get(str(el)))
    #print(roles_l)
    c = Counter(roles_l)
    d = dict(sorted(c.items()))
    pr_list =[]
    for el in d.items():
        pr_list.append(el[0])
    print(pr_list)
    df = pd.DataFrame({'Transaction': new_data_transactions,'Role': roles_l,'Query_Vector': new_data_vectors})
    df.to_csv(path_result,sep='\t', index=False)


def role_determinant(combo_list, type_list, trans_num):
    c = Counter(combo_list)
    d = dict(sorted(c.items()))
    combo_list.clear()
    for el in d.items():
        combo_list.append(el[0])
    combo_list.sort()
    t = Counter(type_list)
    z = dict(sorted(t.items()))
    type_list.clear()
    for i in range(1, 5):
        if z.get(i, 0) == 0:
            type_list.append(0)
        else:
            type_list.append(i)
    #print('combo list',combo_list)
    max = 0
    prefer_role =0
    for j in range(0,len(frames)):
        #print(frames[j],' number ',j)
        count = len(list(set(combo_list)&set(frames[j])))
        #print(count)
        if count>max:
            max = count
            prefer_role=tpce_roles.get(str(j))
        # if combo_list== frames[j]:
        #     roles.update({str(trans_num): tpce_roles.get(str(j))})
        #     print('SUCCESS',trans_num)
        # else:
        #     print('ERROR transaction',trans_num)
    #print('prefer role ',prefer_role)
    roles.update({str(trans_num): tpce_roles.get(str(prefer_role))})
    # m = np.zeros(34, dtype=int)
    # for el in combo_list:
    #     m[tables.get(el)] = 1
    # # Приоритет по первой строке. По максимуму второй берем по индексу элементы из первой,находим там максимум(ы).
    # # Если много максимумов, приоритет на полностью совпадающие type
    #
    # result_1 = transaction_matrix - m
    # judge_array = (result_1 == 0).sum(1)
    #
    # result_2 = type_matrix - type_list
    # # judge_array_2 = (result_2>=0).sum(1)
    # judge_array_2 = (result_2 == 0).sum(1)
    # mat = np.array([judge_array_2, judge_array])
    # max_1 = max(mat[0])
    # ind, = np.where(mat[0] == max_1)
    # best_arr = mat.copy()
    # num = best_arr.shape[1]
    # for i in range(0, num):
    #     if i not in ind:
    #         best_arr[0, i] = 0
    #         best_arr[1, i] = 0
    # max_2 = max(best_arr[1])
    # idxs, = np.where(best_arr[1] == max_2)
    # same_trans = []
    # if len(idxs) > 1:
    #     for j in idxs:
    #         if np.array_equal(type_list, type_matrix[j]):
    #             same_trans.append(j)
    #     if len(same_trans) > 1:
    #         r = random.choice(same_trans)
    #         roles.update({str(trans_num): int(r)})
    #
    #     if len(same_trans) == 1:
    #         roles.update({str(trans_num): int(same_trans[0])})
    #
    #     if not roles.get(str(trans_num), False):
    #         r = random.choice(idxs)
    #         roles.update({str(trans_num): int(r)})
    #
    # if len(idxs) == 1:
    #     roles.update({str(trans_num): int(idxs[0])})


# def role_distributor(data, roles_dict):
#     for index, rows in data.iterrows():
#         rows['role'] = roles_dict.get(str(rows['transaction']))


def SQL_CMD(query):
    sql_cmd = [0, 0]
    parsed = sqlparse.parse(query)[0]
    type = parsed.tokens[0].value
    sql_cmd[0] = query_mode[str(type)][0]
    sql_cmd[1] = len(query)
    return sql_cmd


def PROJ_REL(query):
    query = query.replace(" FORCE INDEX(PRIMARY)", "")
    # print(query)
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    proj_rel = [0, 0]
    arr_string = []
    t_list = []
    new_list = []
    arr_string.append('0' * NUM_TABLES)
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
        d = c.get_identifiers()
        e = list(d)
        proj_rel[0] = len(e)
        tokenizer = RegexpTokenizer(r'\w+')
        S = re.compile(r'FROM([^"]*)WHERE')
        t = S.findall(query)
        tokens = tokenizer.tokenize(str(t))
        for elem in tokens:
            if tables.get(str(elem)):
                t_list.append(elem)

        for el in t_list:
            el = str(el).split(' ')
            indx = tables.get(str(el[0]).lower())
            if indx:
                arr_string[-1] = arr_string[-1][:indx] + '1' + arr_string[-1][indx + 1:]

    #print(arr_string)
    arr_string[-1] = arr_string[-1][::-1]
    proj_rel[1] =[int(i) for i in arr_string[0]]
    #print(proj_rel)
    #proj_rel[1] = arr_string[0]
    #proj_rel[1] = proj_rel[1][::-1]

    #proj_rel[1] = list(proj_rel[1])
    #print(proj_rel)
    #proj_rel[1] = int(proj_rel[1], 2)#####################################################################################
    #print('proj rel', t_list)
    return proj_rel#, t_list #list(map(str, new_list))


def PROJ_ATTR(query):
    proj_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]

    t = parsed.tokens[0].value
    attributes = []
    ilist = []
    if t == 'INSERT':
        res = re.findall(r'\((.*?)\)', str(parsed.tokens[query_mode[t][2]]))
        ilist = str(res).split(',')

    else:
        if t == 'DELETE':
            table_string2 = []
            t_list = []
            table_string2.append('0' * NUM_TABLES)
            proj_attr[1] = [int(item) for item in table_string2[0]]
            table_vector3 = np.zeros(NUM_TABLES, int)  ###
            proj_attr[2] = list(table_vector3)
            return proj_attr#, t_list
        if t == 'SELECT':
            tokenizer = RegexpTokenizer(r'\w+')
            S = re.compile(r'SELECT([^"]*)FROM')
            t = S.findall(query)
            ilist = tokenizer.tokenize(str(t))
        else:
            attributes = parsed.tokens[query_mode[t][2]]
            ilist = list(IdentifierList(TokenList(attributes)).get_identifiers())

    count_list = []
    copy_ilist = ilist.copy()
    new_list = []
    for el in copy_ilist:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(el))
        for elem in tokens:
            if fields.get(str(elem)):
                new_list.append(elem)
    d = Counter(new_list)
    ilist.clear()
    for el in d.items():
        ilist.append(el[0])

    for el in ilist:
        count_list.append(fields[str(el)][0])
    proj_attr[0] = len(ilist)
    c = Counter(count_list)
    d = dict(sorted(c.items()))
    t_list = []

    for el in d.items():
        t_list.append(el[0])

    table_string2 = np.zeros(NUM_TABLES, int)
    table_vector3 = np.zeros(NUM_TABLES, int)  ###
    arr_string = []
    for tab in t_list:
        arr_string.append('0' * MAX_COLUMNS)
        for el in ilist:
            if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                tab_idx = tables[tab]  ###
                table_string2[tab_idx] += 1
                idx = fields[str(el)][1]
                arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

        arr_string[-1] = arr_string[-1][::-1]
        table_vector3[tables[tab]] = int(arr_string[-1], 2)

    proj_attr[1] = table_string2
    proj_attr[2] = table_vector3
    proj_attr[2] = list(proj_attr[2])
    #print('proj attr',t_list)
    return proj_attr#, t_list


def SEL_ATTR(query):
    sel_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    idx = 0
    f_list = []
    t_list = []
    arr = []
    for i in range(0, len(parsed.tokens)):
        if (str(type(parsed.tokens[i])) == '<class \'sqlparse.sql.Where\'>'):
            idx = i

    table_string2 = np.zeros(NUM_TABLES, int)
    table_vector3 = np.zeros(NUM_TABLES, int)  ###

    if idx != 0:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        if len(f_list) > 0:
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            sel_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])
            c = Counter(t_list)
            d = dict(sorted(c.items()))
            # print(d)
            t_list.clear()
            for el in d.items():
                t_list.append(el[0])
                arr.append(el[1])

            idx = 0
            arr_string = []
            for tab in t_list:
                arr_string.append('0' * MAX_COLUMNS)
                for el in f_list:
                    if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                        tab_idx = tables[tab]  ###
                        table_string2[tab_idx] += 1
                        idx = fields[str(el)][1]
                        arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                arr_string[-1] = arr_string[-1][::-1]  ###
                table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

    sel_attr[1] = table_string2
    sel_attr[2] = table_vector3
    sel_attr[2] = list(sel_attr[2])
    #print('select attr',t_list)
    return sel_attr#, t_list


def ORDER_ATTR(query):
    order_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    idx = 0
    f_list = []
    t_list = []
    for i in range(0, len(parsed.tokens)):
        if (str(parsed.tokens[i].value) == "ORDER BY"):
            idx = i + 2

    table_vector3 = np.zeros(NUM_TABLES, int)  ###
    table_string2 = np.zeros(NUM_TABLES, int)

    if idx != 0:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        if len(f_list) > 0:
            # print(f_list)
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            order_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])

            c = Counter(t_list)
            d = dict(sorted(c.items()))
            t_list.clear()
            for el in d.items():
                t_list.append(el[0])

            idx = 0
            arr_string = []
            for tab in t_list:
                arr_string.append('0' * MAX_COLUMNS)
                for el in f_list:
                    if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                        tab_idx = tables[tab]  ###
                        table_string2[tab_idx] += 1
                        idx = fields[str(el)][1]
                        arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                arr_string[-1] = arr_string[-1][::-1]  ###
                table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

    order_attr[1] = table_string2
    order_attr[2] = table_vector3
    order_attr[2] = list(order_attr[2])
    #print('order attr',t_list)
    return order_attr


def GRPBY_ATTR(query):
    grpby_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    idx = 0
    f_list = []
    t_list = []
    for i in range(0, len(parsed.tokens)):
        if (str(parsed.tokens[i].value) == "GROUP BY"):
            idx = i + 2

    table_string2 = np.zeros(NUM_TABLES, int)
    table_vector3 = np.zeros(NUM_TABLES, int)  ###

    if idx != 0:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        if len(f_list) > 0:
            # print(f_list)
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            grpby_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])

            c = Counter(t_list)
            d = dict(sorted(c.items()))
            t_list.clear()
            for el in d.items():
                t_list.append(el[0])

            idx = 0
            arr_string = []
            for tab in t_list:
                arr_string.append('0' * MAX_COLUMNS)
                for el in f_list:
                    if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                        tab_idx = tables[tab]  ###
                        table_string2[tab_idx] += 1
                        idx = fields[str(el)][1]
                        arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                arr_string[-1] = arr_string[-1][::-1]  ###
                table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

    grpby_attr[1] = table_string2
    grpby_attr[2] = table_vector3
    grpby_attr[2] = list(grpby_attr[2])
    #print('group',t_list)
    return grpby_attr


def VALUE_CTR(query):
    value_ctr = [0, 0, 0, 0, 0]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    copy_query = query
    strings = re.compile(r'[\'\"]([^\'\"]+)[\'\"][,\s)]?')
    strings_list = strings.findall(copy_query)
    if (len(strings_list)) > 0:
        value_ctr[0] = len(strings_list)
        s = "".join(strings_list)
        value_ctr[1] = len(s)
        # print(1)
        for el in strings_list:
            copy_query = copy_query.replace(str(el), "xxx")

    copy_query = copy_query.replace("LIMIT ", "xxx")

    numeric = re.compile(r'[\s(]([\d.]+)[,\s)]?')
    numeric_list = numeric.findall(copy_query)
    if len(numeric_list) > 0:
        value_ctr[2] = len(numeric_list)

    joins = re.compile(r'\b(JOIN)\b')
    joins_list = joins.findall(copy_query)
    if len(joins_list) > 0:
        value_ctr[3] = len(joins_list)

    ands = re.compile(r'\b(AND|OR)\b')
    ands_list = ands.findall(copy_query)
    if len(ands_list) > 0:
        value_ctr[4] = len(ands_list)

    return (value_ctr)


def classic_vector_extractor(query):
    sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr = feature_extractor(query)
    Q = np.hstack([sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr])
    #print(time.time())
    return Q ##, combo_list

def short_vector_extractor(query):
    sql_cmd, proj_rel_dec, proj_attr_dec,  value_ctr = short_feature_extractor(
        query)
    Q = np.hstack([sql_cmd, proj_rel_dec, proj_attr_dec, value_ctr])
    return Q


def feature_extractor(query):
    sql_cmd = np.hstack(SQL_CMD(query))
    proj_attr= PROJ_ATTR(query)
    proj_attr_dec = np.hstack(proj_attr)
    proj_rel = PROJ_REL(query)
    proj_rel_dec = np.hstack(proj_rel)
    sel_attr= SEL_ATTR(query)
    sel_attr_dec = np.hstack(sel_attr)
    order_attr_dec = np.hstack(ORDER_ATTR(query))
    grpby_attr_dec = np.hstack(GRPBY_ATTR(query))
    value_ctr = np.hstack(VALUE_CTR(query))

    #combo_list = t_list1 + t_list2 + t_list3

    return sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr#, combo_list

def short_feature_extractor(query):
    sql_cmd = np.hstack(SQL_CMD(query))
    proj_attr, t_list2 = PROJ_ATTR(query)
    proj_attr_dec = np.hstack(proj_attr)
    proj_rel, t_list1 = PROJ_REL(query)
    proj_rel_dec = np.hstack(proj_rel)
    value_ctr = np.hstack(VALUE_CTR(query))

    return sql_cmd, proj_rel_dec, proj_attr_dec ,value_ctr

def cut_query_from_log(path, path_result):
    data = pd.read_csv(path, sep="\t", header=None)
    print(len(data.columns))
    print(data)
    data.columns = ["del1", "transaction", "query","del2"]
    #data.columns = ["del1", "transaction", "query"]
    print(data.loc[0])
    data.drop('del1', axis=1, inplace=True)
    data.drop('del2', axis=1, inplace=True)
    transaction_numbering(data)
    # print(data)
    data['query'].replace('', np.nan, inplace=True)
    data.dropna(subset=['query'], inplace=True)
    data.insert(1, "role", 0)

    # for index, rows in data.iterrows():
    #     print(index)
    #     if rows['query'] in tabu_list:
    #         data.drop([index],inplace=True)
    #     if data.loc[index]['query'] == 'rollback':
    #         data.loc[index]["query"] = 'commit'
    # print(data)
    print('writing')
    data.to_csv(path_result, sep='\t',index=False, header=False)


def make_anomalies(path, path_result, percent):
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ["transaction", 'role', "query"]
    data.drop([0], inplace=True)
    # print(data[-1:]['transaction'])
    count = (int(data[-1:]['transaction']) * percent) // 100
    # print(count)
    # print('Количество аномалий(идут первыми) :',count)
    all_roles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    empty = 0
    for i in range(1, count + 1):
        try:
            r = data.loc[data['transaction'] == str(i), 'role'].values[0]
            c = all_roles.copy()
            c.remove(int(r))
            anomaly = random.choice(c)
            data.loc[(data['transaction'] == str(i)), 'role'] = str(anomaly)
        except:
            empty = empty + 1
    # print('Количество аномалий(идут первыми) :', count-empty)

    data.to_csv(path_result, sep='\t',index=False, header=False)
    f = open('data\\test\\anomaly_count.txt', 'w')
    f.write(str(percent))
    f.write("\n")
    f.write(str(count - empty))
    f.close()


def make_classic_vectors(source,result):
    start_time = time.time()
    data = pd.read_csv(source, sep="\t", header=None)
    data.columns = ["transaction", 'role', "query"]
    print(data)
    data.drop(data[data['query'] =='commit'].index,inplace=True)
    data.reset_index()
    data['query'] = data['query'].apply(lambda x: classic_vector_extractor(x).tolist())
    d = data.to_dict('records')
    f = open(result,'w',encoding='utf-8')
    json.dump(d,f,ensure_ascii=False)
    end_time = (time.time() - start_time)
    print(result,"\t",end_time)
    print(f'Done {result}')


def make_transaction_vectors(source, result):
    start_time = time.time()
    data = pd.read_csv(source, sep="\t", header=None)
    data.columns = ["transaction", 'role', "query"]
    data.drop(data[data['query'] == 'commit'].index, inplace=True)
    data.reset_index()
    new_list =[]
    role_list_new =[]

    data['query'] = data['query'].apply(lambda x: classic_vector_extractor(x).tolist())
    for i in range (1,data.iloc[-1]['transaction']+1):
        trans_num=i
        new_vector = []
        role = 0
        queries = data.loc[data['transaction'] == i]
        for index, rows in queries.iterrows():
            new_vector= new_vector+list(rows['query'])
            role = rows['role']
            ######################################## 6042 нужно поменять
        ext= 6180 - len(new_vector)
        new_vector = new_vector+[0]*ext
        new_list.append(new_vector)
        role_list_new.append(role)

    data = pd.DataFrame(list(zip(role_list_new,new_list)),columns =['role','query'])

    d = data.to_dict('records')
    f = open(result, 'w', encoding='utf-8')
    json.dump(d, f, ensure_ascii=False)
    end_time = (time.time() - start_time)
    print(result, "\t", end_time)
    print(f'Done {result}')

if __name__ == '__main__':
    #Вектор 309
    #максимальная длина 20
    #6180

    #make_classic_vectors(source_data_train,result_data_train_classic)
    #make_classic_vectors(source_data_test, result_data_test_classic)
    #make_classic_vectors(source_data_valid,result_data_valid_classic)

    make_transaction_vectors(source_data_train, result_data_train_transaction)
    make_transaction_vectors(source_data_test, result_data_test_transaction)
    make_transaction_vectors(source_data_valid, result_data_valid_transaction)


