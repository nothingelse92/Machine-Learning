import csv
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet
from operator import itemgetter,attrgetter
import gensim
from gensim.models import Word2Vec
import shutil


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

def wordnet_wup_sim(csv_path):
	out_file_name = "wordnet_result_" + csv_path
	wordpairs = list(csv_parser(csv_path))
	print wordpairs[0]
	wordpairs = cal_rank(wordpairs,3)
	ans_list = []
	rank = 0 
	pre_value = -100
	sum_sim = 0
	not_exsist = []
	pair_id = 0
	for wordpair in wordpairs:
		fst_word = wordpair[1]
		sec_word = wordpair[2]
		sim_float = wordpair[3]
		fst_synsets = wordnet.synsets(fst_word)
		sec_synsets = wordnet.synsets(sec_word)
		max_sim = -10
		for fst_syn in fst_synsets:
			for sec_syn in sec_synsets:
				try:
					if fst_syn.pos() == sec_syn.pos():
						max_sim = max(max_sim,fst_syn.wup_similarity(sec_syn))
				except Exception ,e :
					print "%s :  %s\n" % (fst_syn,sec_syn)
					print str(e) + '\n'	
		if max_sim != -10:
			sum_sim += max_sim
		if max_sim == -10:
			not_exsist.append(pair_id)
		wordpair.append(max_sim)
		ans_list.append(wordpair)
		pair_id += 1

	for t_id in not_exsist:
		ans_list[t_id][5] = sum_sim/(len(wordpairs) - len(not_exsist))
	
	ans_list = cal_rank(ans_list,5)
	num = 0
	sum_gap = 0
	for line in ans_list:
		num += 1
		sum_gap += (line[4] - line[6])*(line[4] - line[6])
	print num
	output_file(out_file_name,ans_list)
	return (1-sum_gap*6.0/(num*(num*num - 1)))
	

def word_2_vec():
	csv_paths = ['set1.csv','set2.csv','combined.csv']
	model = Word2Vec.load_word2vec_format('/root/libanghuai/homework/GoogleNews-vectors-negative300.bin', binary=True)
	for csv_path in csv_paths:
		print "deal_with %s \n" % csv_path
		out_file_name = "word2vec_result_"+csv_path
		wordpairs = list(csv_parser(csv_path))
		wordpairs = cal_rank(wordpairs,3)
		ans_list=[]
		for wordpair in wordpairs:
			fst_word = wordpair[1]
			sec_word = wordpair[2]
			max_sim = model.similarity(fst_word,sec_word)
			wordpair.append(max_sim)
			ans_list.append(wordpair)

		ans_list = cal_rank(ans_list,5)
		num = 0
		sum_gap = 0
		for line in ans_list:
			num += 1
			sum_gap += (line[4] - line[6])*(line[4] - line[6])
		print num
		output_file(out_file_name,ans_list)
		print (1-sum_gap*6.0/(num*(num*num - 1)))

def pre_deal(infile, outfile, line):
	with open(infile, 'r') as old:
		with open(outfile, 'w') as new:
			new.write(str(line) + "\n")
			shutil.copyfileobj(old, new)

def glove():
	csv_paths = ['set1.csv','set2.csv','combined.csv']
	glove_file="glove.twitter.27B.100d.txt"
	_,_,tokens,dimensions,_ = glove_file.split('.')
	print tokens + "\n"
	print dimensions
	num_lines = 1193514 
	dims = int(dimensions[:-1])
	gensim_file='glove_model.txt'
	gensim_first_line = "{} {}".format(num_lines, dims)
	pre_deal(glove_file, gensim_file, gensim_first_line)
	model=gensim.models.Word2Vec.load_word2vec_format(gensim_file,binary=False)
	for csv_path in csv_paths:
		print "deal_with %s \n" % csv_path
		out_file_name = "glove_result_"+csv_path
		wordpairs = list(csv_parser(csv_path))
		wordpairs = cal_rank(wordpairs,3)
		ans_list=[]
		for wordpair in wordpairs:
			fst_word = wordpair[1].lower()
			sec_word = wordpair[2].lower()
			try:
				max_sim = model.similarity(fst_word,sec_word)
			except Exception,e:
				print fst_word + ' ' + sec_word+'\n'
				max_sim = 0.5
				print str(e)+'\n'
			wordpair.append(max_sim)
			ans_list.append(wordpair)

		ans_list = cal_rank(ans_list,5)
		num = 0
		sum_gap = 0
		for line in ans_list:
			num += 1
			sum_gap += (line[4] - line[6])*(line[4] - line[6])
		print num
		output_file(out_file_name,ans_list)
		print (1-sum_gap*6.0/(num*(num*num - 1)))

if __name__ == "__main__":
	csv_paths = ['set1.csv','set2.csv','combined.csv']
	for csv_path in csv_paths:
		print wordnet_wup_sim(csv_path)
