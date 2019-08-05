from py_formatted_log import *


def gen_legal_pinyin_list(msg):
    lst = msg.split() if type(msg) == str else msg
    word_max_len = max([len(i) for i in lst])
    legal_pinyin_list = [set() for i in range(word_max_len + 1)]
    for word in lst:
        for x in range(1, len(lst)):
            if len(word) == x:
                legal_pinyin_list[x].add(word)
    return legal_pinyin_list


def stat_pos(msg, char_list):
    legal_pos_dict = dict(
        zip(char_list, [set() for x in range(len(char_list))]))
    legal_pinyin_list = gen_legal_pinyin_list(msg)
    for letter in char_list:
        for i in legal_pinyin_list:
            for j in i:
                if j.find(letter) != -1:
                    legal_pos_dict[letter].add((len(j), j.find(letter) + 1))
    return (legal_pos_dict, legal_pinyin_list)


def main():
    lower_case = [chr(i) for i in range(97, 123)]
    with open('data/all_legal_pinyin.txt', 'r', encoding="utf-8") as fp:
        all_legal_pinyin = fp.read()
    # infolist(all_legal_pinyin.split(), "all_legal_pinyin", 7)

    # 统计正确拼音中每个字母的位置
    legal_pos_dict, legal_pinyin_list = stat_pos(all_legal_pinyin, lower_case)

    upper_case = [chr(i) for i in range(65, 91)]
    fname = "留言1-1_op_已加密_转写"
    with open("src/" + fname + ".txt", "r", encoding="utf-8") as fp:
        article = fp.read()
        total_chars = len(article.replace(" ", ""))
        total_words = len(article.split())

    # 统密文中每个字母的位置
    occ_pos_dict, occ_pinyin_list = stat_pos(article, upper_case)

    # 由字母位置得出，是密文的最大嫌疑范围，逐步缩窄
    legal_dict = dict(zip(lower_case, [set() for i in range(26)]))
    # 留个备份，以免误操作
    legal_dict_copy = dict(zip(lower_case, [set() for i in range(26)]))
    # 绝对排除嫌疑
    never_dict = dict(zip(lower_case, [set() for i in range(26)]))

    # 如果密文中符号的位置的集合，正好是某字母正确位置的子集，那它可能是这个字母；
    # 反之如果不在该字母位置集合的范围内，它就绝对不会是这个字母
    for i, j in occ_pos_dict.items():
        for k, l in legal_pos_dict.items():
            if j.issubset(l):
                legal_dict[k] = legal_dict[k].union(i)
                legal_dict_copy[k] = legal_dict[k].union(i)
            else:
                never_dict[k] = never_dict[k].union(i)

    infodict(legal_dict, "legal_dict_0", 1, 13)
    infodict(never_dict, "never_dict_0", 1, 13)

    has_single_pos = [i for i in legal_pos_dict.keys() if max([j[1] for j in legal_pos_dict[i]]) < 2]
    has_multi_pos = list(set(lower_case) - set(has_single_pos))
    has_single_pos_set = set()
    has_multi_pos_set = set()
    for i in occ_pos_dict.keys():
        if max([j[1] for j in occ_pos_dict[i]]) < 2:
            has_single_pos_set.add(i)
        else:
            has_multi_pos_set.add(i)

    # 位置多变的符号一定不会是位置单一的字母
    for i in has_multi_pos:
        legal_dict[i] = legal_dict[i] - has_single_pos_set
        # if total_chars > 400:
        #     never_dict[i] = never_dict[i].union(has_single_pos_set)
    for i in has_single_pos:
        legal_dict[i] = legal_dict[i] - has_multi_pos_set
        # if total_chars > 400:
        #     never_dict[i] = never_dict[i].union(has_multi_pos_set)

    infolist(legal_pinyin_list, "legal_pinyin_list", 1, 7)
    infolist(occ_pinyin_list, "occ_pinyin_list", 1, 7)
    infodict(legal_pos_dict, "legal_pos_dict", 1, 6)
    infodict(occ_pos_dict, "occ_pos_dict", 1, 6)
    infosimple(has_single_pos,"has_single_pos",20)
    infosimple(has_single_pos_set,"has_single_pos_set",20)
    infosimple(has_multi_pos,"has_multi_pos",20)
    infosimple(has_multi_pos_set,"has_multi_pos_set",20)
    infodict(legal_dict, "legal_dict", 1, 13)
    infodict(never_dict, "never_dict", 1, 13)

main()
