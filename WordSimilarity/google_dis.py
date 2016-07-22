import csv
#from nltk.corpus import wordnet_ic
#from nltk.corpus import wordnet
from operator import itemgetter,attrgetter
from selenium import webdriver
import re
import math
import time

def csv_parser(csv_path):
	csv_reader = csv.reader(open(csv_path,'rb'))
	num = 0
	for line in csv_reader:
		if num == 0:
			num = 1
			continue
		yield [num,line[0],line[1],float(line[2])]
		num += 1

def cal_rank(in_list,rank_item):
	in_list.sort(key=itemgetter(rank_item))
	for i in range(len(in_list)):
		in_list[i].append(i+1)

	for i in range(len(in_list)):
		if i > 0:
			if abs(float(in_list[i][rank_item]) - float(in_list[i-1][rank_item])) < 0.0000001:
				s = i-1
				rank_sum = 0
				while i < len(in_list) and abs(float(in_list[i][rank_item]) - float(in_list[i-1][rank_item])) < 0.0000001:
					i += 1
				t = i-1
				for j in range(s,t+1):
					rank_sum += in_list[j][rank_item+1]
				for j in range(s,t+1):
					in_list[j][rank_item+1] = rank_sum*1.0/(t-s+1)
	return in_list

def output_file(file_name,out_list):
	file_out = open(file_name,'wb')
	out_list.sort(key=itemgetter(0))
	for line in out_list:
		i = 0
		for word in line:
			if i == 0:
				i = 1
				file_out.write(str(word))
				continue
			file_out.write(","+str(word))
		file_out.write('\n')
	file_out.close()

def get_word_num(word):
        browser = webdriver.PhantomJS()
        browser.get('https://www.google.com/search?hl=en&q=%s' % word) 
        content = browser.find_element_by_id('resultStats').text
        pattern = re.compile(r'\d+')
        result = pattern.findall(content)
	temp_result=""
	for item in result:
		temp_result += item
        num = int(temp_result)
        return num

def google_dis(csv_path):
	out_file_name = "googld_distance_result_" + csv_path
        wordpairs = list(csv_parser(csv_path))
        print wordpairs[0]
        wordpairs = cal_rank(wordpairs,3)
	ans_list = []
	N = math.log(25270000000,2)# the search result of word 'a'
	for wordpair in wordpairs:
		fst_word = wordpair[1]
		sec_word = wordpair[2]
		try:
			fst_num = math.log(get_word_num(fst_word),2)
			sec_num = math.log(get_word_num(sec_word),2)
			all_num = math.log(get_word_num(fst_word+" "+sec_word),2)
		except Exception,e:
			print "fst_word: %s sec_word: %s combined_word: %s \n" % (fst_word,sec_word,fst_word + " " + sec_word)
			print str(e) + '\n'
		google_sim = (max(fst_num,sec_num) - all_num)/(N - min(fst_num,sec_num)) 
		wordpair.append(google_sim)
		ans_list.append(wordpair)

	ans_list = cal_rank(ans_list,5)
    num = 0
    sum_gap = 0
    for line in ans_list:
            num += 1
            sum_gap += (line[4] - line[6])*(line[4] - line[6])
    print num
    output_file(out_file_name,ans_list)
    return (1-sum_gap*6.0/(num*(num*num - 1)))
if __name__ == "__main__":
	csv_paths = ['set1.csv']
	for csv_path in csv_paths:
		print google_dis(csv_path)

