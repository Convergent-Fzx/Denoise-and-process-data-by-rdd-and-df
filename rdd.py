from pyspark import SparkContext, SparkConf
import re
import itertools

class Project2:
    def is_valid_word(self, word):
        return len(word) >= 1 and 'a' <= word[0] <= 'z' and word not in self.stopwords_broadcast.value

    def run(self, inputPath, outputPath, stopwordsPath, k):
        conf = SparkConf().setAppName("project2_rdd").setMaster("local")
        sc = SparkContext(conf=conf)
        
        stopwords_list = sc.textFile(stopwordsPath).collect()
        self.stopwords_broadcast = sc.broadcast(set(stopwords_list))
        
        lines = sc.textFile(inputPath)
        
        '''提取第一个单词和剩余部分'''
        lines = lines.map(lambda line: (line.split(',')[0].lower(), [word.lower() for word in line.split(',')[1:]]))
              
        '''过滤和处理有效单词'''
        validLines = lines.map(
            lambda line: (
                line[0], 
                [word for word in re.split("[\\s*$&#/\"'\\,.:;?!\\[\\](){}<>~\\-_]+", ' '.join(line[1])) if self.is_valid_word(word)]
            )
        )
        
        '''计算非法行数'''
        invalid_lines = validLines.map(lambda x: (x[0], (1, len(x[1])))).filter(lambda x: x[1][1] < 3)
        res_inv = invalid_lines.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        res_inv = res_inv.map(lambda x:(x[0],x[1][0]))
        res_inv = res_inv.sortByKey()
        '''计算总行数'''
        valid_lines = validLines.filter(lambda x: len(x[1]) >= 3)
        line_zong = valid_lines.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y)
        '''line_zong = line_zong.map(lambda x:((x[0],''),x[1]))'''
        
        '''计算三元组'''
        sanyuanzu = validLines.flatMap(
            lambda x: [(x[0], tuple(sorted(triplet))) for triplet in itertools.combinations(x[1], 3)]
        ).map(lambda x: (x, 1))
        sanyuanzu_res = sanyuanzu.reduceByKey(lambda x, y: x + y)
        sanyuanzu_res = sanyuanzu_res.map(lambda x:((x[0][0]),(x[0][1],x[1])))
        
        k = int(k)
        '''计算三元组频率比例'''
        joined_data = sanyuanzu_res.join(line_zong)
        bili = joined_data.map(lambda x: ((x[0],x[1][0][0]), x[1][0][1] / x[1][1]))
        bili = bili.map(lambda x:(x[0][0],(x[0][1],x[1])))
        bili = bili.groupByKey().flatMap(lambda x: [(x[0], item) for item in sorted(x[1], key=lambda y: -y[1])[:k]])
        bili = bili.sortByKey()

        join_res = res_inv.join(bili)
        '''非法行计算'''
        res_inv_formatted = res_inv.map(lambda x: "{}\tinvalid line:{}".format(x[0], x[1]))
        
        formatted_res = join_res.flatMap(lambda x: [
            "{}\t{}:{}".format(x[0], ','.join(x[1][1][0]), x[1][1][1])
        ])

        '''合并结果'''
        combined_results = res_inv_formatted.union(formatted_res)

        
        combined_results = combined_results.sortBy(lambda x:x[0])
        combined_results.coalesce(1).saveAsTextFile(outputPath)
        
        sc.stop()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Wrong arguments")
        sys.exit(-1)
    Project2().run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

