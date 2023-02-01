from collections import Counter

import numpy as np
import pandas as pd
import re
import regex
import sqlparse
from sklearn.cluster import KMeans
from sqlparse.sql import IdentifierList, TokenList, Parenthesis
from nltk.tokenize import RegexpTokenizer

MAX_COLUMNS = 30
NUM_TABLES = 34
path = 'C:\\Users\\user\\Desktop\\query_logs\\processing\\raw_logs.txt'
path_result = 'C:\\Users\\user\\Desktop\\query_logs\\processing\\cutted_queries.csv'
path_Hex_vector ='C:\\Users\\user\\Desktop\\query_logs\\processing\\hex_vectors.csv'
tables = {'account_permission': 0, 'address': 1, 'broker': 2, 'cash_transaction': 3, 'charge': 4, 'commission_rate': 5,
          'company': 6, 'company_competitor': 7, 'customer': 8,
          'customer_account': 9, 'customer_taxrate': 10, 'daily_market': 11, 'exchange': 12, 'financial': 13,
          'holding': 14, 'holding_history': 15, 'holding_summary': 16,
          'industry': 17, 'last_trade': 18, 'news_item': 19, 'news_xref': 20, 'sector': 21, 'security': 22,
          'seq_trade_id': 23, 'settlement': 24, 'status_type': 25, 'taxrate': 26,
          'trade': 27, 'trade_history': 28, 'trade_request': 29, 'trade_type': 30, 'watch_item': 31, 'watch_list': 32,
          'zip_code': 33}

# [0]- тип запроса [1] - индекс листа с таблицами [2] - индекс листа с полями
query_mode = {'SELECT': [1, 6, 2], 'INSERT': [2, 4, 4], 'UPDATE': [3, 2, 6], 'DELETE': [4, 4]}
tabu_list = ['SELECT LAST_INSERT_ID()', 'SET TRANSACTION ISOLATION LEVEL READ COMMITTED',
             'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ', 'SET TRANSACTION ISOLATION LEVEL SERIALIZABLE',
             'rollback']

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


# Возможно один элемент это матрица. где каждая строка этом массив использованных таблиц для селект,инсерт, делит.

# РАСКОМЕНТИТЬ В ИТЕРАТОРЕ FOR, ЧТОБЫ ПО ВСЕМ ПЕРЕБИРАЛ

# Заметка про МЕТОД1. Там у нас есть еще одна таблица y test с правильными значениями.
# И после классификации получившийся y_test сравнивается с ней.

# ЗАМЕТКА.  ВСЕ процедуры с векторами надо переделать, чтобы второй массив включал в себя все таблицы, а не только использованные.

def transaction_numbering(data):
    num = 0
    last = data.last_valid_index()

    for index, rows in data.iterrows():
        if(index!=last):
            if (data.loc[index]['query'] == 'commit'):
                if(data.loc[index+1]['query'] != 'commit'):
                    num = num + 1
                else:
                    data.drop([index])
        rows['transaction'] = num
        print(rows['transaction'])
    #return data

def transaction_iterator(data):
    count = data['transaction'].max()
    print(count)

    ############ НА ВРЕМЯ
    count=10

    new_data_list = []
    #tabu_list =['commit','SELECT LAST_INSERT_ID()','SET TRANSACTION ISOLATION LEVEL READ COMMITTED','SET TRANSACTION ISOLATION LEVEL REPEATABLE READ','SET TRANSACTION ISOLATION LEVEL SERIALIZABLE','rollback']
    for i in range (1,count):
        queries =data.loc[data['transaction']==i]
        vector = ""
        for index,rows in queries.iterrows():
            if rows['query'] not in tabu_list and not "commit":
                print('QUERY',i,index,rows['query'])
                #if(i==3 & index==46): #удалить нахрен условие
                vector = vector +str(query_preprocessing(rows['query']))+" "
        # s = " ".join(vector)
        # s = s.replace(" ", "")
        # s = s.replace("[", "")
        # s = s.replace("]", "")
        # s = s.replace("\n", "")
        # s = s.replace("@", " ")
        # new_data_list.append(s)
        new_data_list.append(vector)
    print(new_data_list)
    # f= open("C:\\Users\\user\\Desktop\\query_logs\\processing\\data.txt",'w+')
    # for el in new_data_list:
    #     print(el,file=f)
    # f.close()
    #df = pd.DataFrame(new_data_list)
    #df.to_csv(path_Hex_vector, index=False, header=False)


def SQL_CMD(query):
    sql_cmd = [0, 0]
    parsed = sqlparse.parse(query)[0]
    type = parsed.tokens[0].value
    sql_cmd[0] = query_mode[str(type)][0]
    sql_cmd[1] = len(query)
    print(sql_cmd)
    return sql_cmd


def PROJ_REL(query):
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    proj_rel = [0, 0]
    arr_string = []
    arr_string.append('0' * NUM_TABLES)
    parsed = sqlparse.parse(query)[0]
    t = parsed.tokens[0].value
    idx = 0
    for i in range(0, len(parsed.tokens)):
        if (str(type(parsed.tokens[i].value)) == 'FROM'):
            idx=i+2

    if idx!=0:
        proj_rel[0] = len(list(IdentifierList(TokenList(parsed.tokens[idx])).get_identifiers()))
    #print(proj_rel[0])
        for el in IdentifierList(TokenList(parsed.tokens[query_mode[t][1]])).get_identifiers():
        #print(str(el))
            indx = tables.get(str(el))
            if indx:
                arr_string[-1] = arr_string[-1][:indx] + '1' + arr_string[-1][indx + 1:]

    proj_rel[1] = arr_string[0]
    #print(proj_rel[1])
    proj_rel[1] = proj_rel[1][::-1]
    #print(proj_rel[1])
    proj_rel[1] = int(proj_rel[1], 2)
    print(proj_rel)
    return proj_rel


def PROJ_ATTR(query):
    proj_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    #print(parsed.tokens)
    t = parsed.tokens[0].value
    attributes = []
    ilist = []
    if t == 'INSERT':
        res = re.findall(r'\((.*?)\)',str(parsed.tokens[query_mode[t][2]]))
        #print(str(res))
        ilist = str(res).split(',')

        ##
        #ilist = list(ilist.get_identifiers())
        ##
    else:
        if t =='DELETE':
            table_string2 = []
            table_string2.append('0' * NUM_TABLES)
            proj_attr[1] = [int(item) for item in table_string2[0]]
            table_vector3 = np.zeros(NUM_TABLES, int)  ###
            proj_attr[2] = list(table_vector3)
            return proj_attr
        else:
            attributes = parsed.tokens[query_mode[t][2]]
            ilist = list(IdentifierList(TokenList(attributes)).get_identifiers())

    count_list = []
    # for el in ilist.get_identifiers():
    copy_ilist = ilist.copy()
    new_list = []
    for el in copy_ilist:
        # if not fields.get(str(el)):
        tokenizer = RegexpTokenizer(r'\w+')
        #print('Обработка', str(el))
        tokens = tokenizer.tokenize(str(el))
        #print('токены', tokens)
        for elem in tokens:
            if fields.get(str(elem)):
                new_list.append(elem)
        # new_list.remove(el)
    d = Counter(new_list)
    ilist.clear()
    for el in d.items():
        ilist.append(el[0])

    for el in ilist:
        count_list.append(fields[str(el)][0])
    # print('ilist', ilist)
    proj_attr[0] = len(ilist)
    c = Counter(count_list)
    d = dict(sorted(c.items()))
    t_list = []

    for el in d.items():
        t_list.append(el[0])

    table_string2 = []  ###
    table_string2.append('0' * NUM_TABLES)  ###
    table_vector3 = np.zeros(NUM_TABLES, int)  ###
    arr_string = []
    for tab in t_list:
        arr_string.append('0' * MAX_COLUMNS)
        for el in ilist:
            if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                tab_idx = tables[tab]  ###
                table_string2[0] = table_string2[0][:tab_idx] + (str(int(table_string2[0][tab_idx]) + 1)) + \
                                   table_string2[0][tab_idx + 1:]  ###
                idx = fields[str(el)][1]
                arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

        arr_string[-1] = arr_string[-1][::-1]
        table_vector3[tables[tab]] = int(arr_string[-1], 2)

    #print('table string', table_string2)
    #print('vector3', table_vector3)
    proj_attr[1] = [int(item) for item in table_string2[0]]  ###
    proj_attr[2] = table_vector3
    proj_attr[2] = list(proj_attr[2])

    print(proj_attr)
    return proj_attr


def SEL_ATTR(query):
    sel_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    #print(parsed.tokens)
    idx = 0
    f_list = []
    t_list = []
    arr = []
    for i in range(0, len(parsed.tokens)):
        if (str(type(parsed.tokens[i])) == '<class \'sqlparse.sql.Where\'>'):
            idx = i

    table_string2 = []  ###
    table_string2.append('0' * NUM_TABLES)  ###
    table_vector3 = np.zeros(NUM_TABLES, int)  ###

    if idx != 0:
        # q = str(parsed.tokens[idx])
        # print(q)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        # print('токены', tokens)
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        if len(f_list) > 0:
            # print(f_list)
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            sel_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])

            # print(t_list)
            c = Counter(t_list)
            d = dict(sorted(c.items()))
            # print(d)
            t_list.clear()
            for el in d.items():
                t_list.append(el[0])
                arr.append(el[1])

            #print(t_list)

            idx = 0
            arr_string = []
            #print(f_list)
            for tab in t_list:
                arr_string.append('0' * MAX_COLUMNS)
                for el in f_list:
                    if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                        tab_idx = tables[tab]  ###
                        table_string2[0] = table_string2[0][:tab_idx] + (str(int(table_string2[0][tab_idx]) + 1)) + \
                                           table_string2[0][tab_idx + 1:]  ###
                        idx = fields[str(el)][1]
                        arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                arr_string[-1] = arr_string[-1][::-1]  ###
                table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

            #print('table string', table_string2)
            #print('vector3', table_vector3)
    sel_attr[1] = [int(item) for item in table_string2[0]]  ###
    sel_attr[2] = table_vector3
    sel_attr[2] = list(sel_attr[2])

    print(sel_attr)

    return sel_attr


# НЕ РАБОТАЕТ С ПЕРЕНАЗНАЧЕННЫМИ ПОЛЯМИ
def ORDER_ATTR(query):
    order_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    #print(parsed.tokens)
    idx = 0
    f_list = []
    t_list = []
    for i in range(0, len(parsed.tokens)):
        # print(i,parsed.tokens[i].value)
        if (str(parsed.tokens[i].value) == "ORDER BY"):
            idx = i + 2

    table_string2 = []  ###
    table_string2.append('0' * NUM_TABLES)  ###
    table_vector3 = np.zeros(NUM_TABLES, int)  ###

    if idx != 0:
        # q = str(parsed.tokens[idx])
        # print(q)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        #print('токены', tokens)
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        #print(f_list)
        if len(f_list) > 0:
            # print(f_list)
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            order_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])

                # print(t_list)
                c = Counter(t_list)
                # print(c)
                d = dict(sorted(c.items()))
                # print(d)
                t_list.clear()
                for el in d.items():
                    t_list.append(el[0])

                # t_list.sort()
                #print(t_list)
                idx = 0
                arr_string = []
                for tab in t_list:
                    arr_string.append('0' * MAX_COLUMNS)
                    for el in f_list:
                        if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                            tab_idx = tables[tab]  ###
                            table_string2[0] = table_string2[0][:tab_idx] + (str(int(table_string2[0][tab_idx]) + 1)) + \
                                               table_string2[0][tab_idx + 1:]  ###
                            idx = fields[str(el)][1]
                            arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                    arr_string[-1] = arr_string[-1][::-1]  ###
                    table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

    order_attr[1] = [int(item) for item in table_string2[0]]  ###
    order_attr[2] = table_vector3
    order_attr[2] = list(order_attr[2])

    print(order_attr)
    return order_attr


def GRPBY_ATTR(query):
    grpby_attr = [0, [0], [0]]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    #print(parsed.tokens)
    idx = 0
    f_list = []
    t_list = []
    for i in range(0, len(parsed.tokens)):
        # print(i,parsed.tokens[i].value)
        if (str(parsed.tokens[i].value) == "GROUP BY"):
            idx = i + 2

    table_string2 = []  ###
    table_string2.append('0' * NUM_TABLES)  ###
    table_vector3 = np.zeros(NUM_TABLES, int)  ###

    if idx != 0:
        # q = str(parsed.tokens[idx])
        # print(q)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(str(parsed.tokens[idx]))
        #print('токены', tokens)
        for elem in tokens:
            if fields.get(str(elem)):
                f_list.append(elem)

        #print(f_list)
        if len(f_list) > 0:
            # print(f_list)
            c = Counter(f_list)
            f_list.clear()
            for el in c.items():
                f_list.append(el[0])

            grpby_attr[0] = len(f_list)
            for el in f_list:
                t_list.append(fields[el][0])

                # print(t_list)
                c = Counter(t_list)
                # print(c)
                d = dict(sorted(c.items()))
                # print(d)
                t_list.clear()
                for el in d.items():
                    t_list.append(el[0])

                # t_list.sort()
                #print(t_list)
                # print(arr)

                idx = 0
                arr_string = []
                for tab in t_list:
                    arr_string.append('0' * MAX_COLUMNS)
                    for el in f_list:
                        if fields.get(str(el)) and tab == str(fields[str(el)][0]):
                            tab_idx = tables[tab]  ###
                            table_string2[0] = table_string2[0][:tab_idx] + (str(int(table_string2[0][tab_idx]) + 1)) + \
                                               table_string2[0][tab_idx + 1:]  ###
                            idx = fields[str(el)][1]
                            arr_string[-1] = arr_string[-1][:idx] + '1' + arr_string[-1][idx + 1:]

                    arr_string[-1] = arr_string[-1][::-1]  ###
                    table_vector3[tables[tab]] = int(arr_string[-1], 2)  ###

    grpby_attr[1] = [int(item) for item in table_string2[0]]  ###
    grpby_attr[2] = table_vector3
    grpby_attr[2] = list(grpby_attr[2])

    print(grpby_attr)
    return grpby_attr


# Мысль больного разума. Он считает цифры в строковых значениях как числа. Как насчет сначала искать строки, потом их реплейсить, а потом искать числа
def VALUE_CTR(query):
    value_ctr = [0, 0, 0, 0, 0]
    query = sqlparse.format(query, reindent=True, keyword_case='upper')
    parsed = sqlparse.parse(query)[0]
    # print(parsed.tokens)

    #print(query)
    copy_query = query
    strings = re.compile(r'[\'\"]([^\'\"]+)[\'\"][,\s)]?')
    strings_list = strings.findall(copy_query)
    #print(strings_list)
    #print(len(strings_list))
    if (len(strings_list)) > 0:
        value_ctr[0] = len(strings_list)
        s = "".join(strings_list)
        #print(s)
        value_ctr[1] = len(s)
        # print(1)
        for el in strings_list:
            copy_query = copy_query.replace(str(el), "xxx")

    copy_query = copy_query.replace("LIMIT ", "xxx")

    #print('copy', copy_query)
    # print(copy_query)
    numeric = re.compile(r'[\s(]([\d.]+)[,\s)]?')
    numeric_list = numeric.findall(copy_query)
    #print(numeric_list)
    if len(numeric_list) > 0:
        value_ctr[2] = len(numeric_list)

    joins = re.compile(r'\b(JOIN)\b')
    joins_list = joins.findall(copy_query)
    #print(joins_list)
    if len(joins_list) > 0:
        value_ctr[3] = len(joins_list)

    ands = re.compile(r'\b(AND|OR)\b')
    ands_list = ands.findall(copy_query)
    if len(ands_list) > 0:
        value_ctr[4] = len(ands_list)

    print(value_ctr)

    return (value_ctr)


def query_preprocessing(query):
    # query = "UPDATE last_trade SET lt_price = 28.31, lt_vol = lt_vol + 325, lt_dts = '2022-12-23 18:14:12' WHERE lt_s_symb = 'FCCY'"
    #query = "INSERT INTO cash_transaction(ct_dts, ct_t_id, ct_amt, ct_name) VALUES('2022-12-23 18:14:12.000000', 200000008727371, 10595.7, 'Limit-Sell 400 shared of COMMON of IES Utilities, Inc.')"
    # query = "SELECT b_name, SUM(tr_qty * tr_bid_price) AS price_sum FROM trade_request, sector, industry, company, broker, security WHERE tr_b_id = b_id AND tr_s_symb = s_symb AND s_co_id = co_id AND co_in_id = in_id AND sc_id = in_sc_id AND b_name IN ('Alfred L. Laurey', 'Leonard J. Ridpath', 'Sylvia P. Stieff', 'Stefania L. Junk', 'Albert V. Medler', 'Albert Z. Titlow', 'Wayne L. Warney', 'Donald A. Gunsolus', 'Carrie G. Sheffey', 'Jo W. Lyng', 'Joseph D. Lofties', 'Vincent U. Tena', 'Michael X. Gramlich', 'Floretta E. Coner', 'Elizabeth O. Peli', 'Chris I. Triveno', 'Mabel G. Clawson', 'Clint G. Lindenpitz', 'Walter M. Attaway', 'David N. Kallenberger', 'Betty K. Hoffner', 'Pedro P. Kovarovic', 'Raymond N. Pullman', 'Juanita K. Reddout') AND sc_name = 'Transportation' GROUP BY b_name ORDER BY price_sum DESC"
    # query = "SELECT ca_id, ca_bal, COALESCE(SUM(hs_qty * lt_price),0) AS price_sum FROM customer_account LEFT OUTER JOIN holding_summary ON hs_ca_id = ca_id, last_trade WHERE ca_c_id = 4300002890 AND lt_s_symb = hs_s_symb GROUP BY ca_id,ca_bal ORDER BY price_sum ASC LIMIT 10"
    sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr = feature_extractor(query)
    Q = make_vector_Q(sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr)
    #print(Q)
    print(len(Q))
    return Q



def feature_extractor(query):
    sql_cmd = np.hstack(SQL_CMD(query))
    proj_rel_dec = np.hstack(PROJ_REL(query))
    proj_attr_dec = np.hstack(PROJ_ATTR(query))
    sel_attr_dec = np.hstack(SEL_ATTR(query))
    order_attr_dec = np.hstack(ORDER_ATTR(query))
    grpby_attr_dec = np.hstack(GRPBY_ATTR(query))
    value_ctr = np.hstack(VALUE_CTR(query))
    return sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr


def make_vector_Q(sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr):
    return np.hstack([sql_cmd, proj_rel_dec, proj_attr_dec, sel_attr_dec, order_attr_dec, grpby_attr_dec, value_ctr])



def cut_query_from_log():
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ["del1", "transaction", "query"]
    data.drop('del1', axis=1, inplace=True)
    # data = data.reset_index()
    # data.drop('transaction', axis=1, inplace=True)
    transaction_numbering(data)
    print(data)
    data['query'].replace('', np.nan, inplace=True)
    data.dropna(subset=['query'], inplace=True)
    data.insert(1, "role", 0)


    for index, rows in data.iterrows():
        if rows['query'] in tabu_list:
            data.drop([index],inplace=True)
    print(data)
    data.to_csv(path_result, index=False, header=False)


if __name__ == '__main__':
    #РАСКОМЕНТИТЬ
    cut_query_from_log()

    # X = np.array([[1, 2,0], [1, 4,45], [1, 0,77],[10, 2,5], [10, 4,11], [10, 0,45]])
    # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    # print(X)
    # #print(kmeans.labels_)
    # #print(kmeans.predict([[0, 0,44], [12, 3,0]]))
    # #print(kmeans.cluster_centers_)



    #####################
    #
    data = pd.read_csv(path_result, sep=",", header=None)
    data.columns = ["transaction",'role', "query"]
    print(data)
    #transaction_iterator(data)
